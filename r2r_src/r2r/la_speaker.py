import torch
import numpy as np
import os
import torch.nn.functional as F
import json
from torch.autograd import Variable

from utils.misc import angle_feature, length2mask
from models import model_speaker

from r2r.parser import args

class LA_Speaker():
    '''
    Decoupled Label Speaker (DLS)
    or
    Landamrk and Action Speaker (LA_Speaker)
    '''
    env_actions = {
        'left': (0,-1, 0), # left
        'right': (0, 1, 0), # right
        'up': (0, 0, 1), # up
        'down': (0, 0,-1), # down
        'forward': (1, 0, 0), # forward
        '<end>': (0, 0, 0), # <end>
        '<start>': (0, 0, 0), # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, tok, listener=None):
        self.env = env
        self.feature_size = args.image_feat_size
        self.tok = tok
        # self.tok.finalize()
        self.listener = listener

        # Model
        print("VOCAB_SIZE", self.tok.vocab_size)
        # args.angle_feat_size = 128
        self.encoder = model_speaker.SpeakerEncoder(self.feature_size+args.angle_feat_size*32, args.rnn_dim, args.dropout, bidirectional=args.bidir).cuda()
        self.decoder = model_speaker.LASpeakerDecoder(self.tok.vocab_size, args.rnn_dim, 0,
                                            args.rnn_dim, args.dropout).cuda()
        # Optimizers
        if args.optim == 'rms':
            optimizer = torch.optim.RMSprop
        elif args.optim == 'adam':
            optimizer = torch.optim.Adam
        elif args.optim == 'adamW':
            optimizer = torch.optim.AdamW
        elif args.optim == 'sgd':
            optimizer = torch.optim.SGD
        else:
            assert False

        self.encoder_optimizer = optimizer(self.encoder.parameters(), lr=args.lr)
        self.decoder_optimizer = optimizer(self.decoder.parameters(), lr=args.lr)

        # Evaluation
        self.softmax_loss = torch.nn.CrossEntropyLoss(ignore_index=0)

        # Will be used in beam search
        self.nonreduced_softmax_loss = torch.nn.CrossEntropyLoss(
            ignore_index=0,#self.tok.word_to_index['<PAD>'],
            size_average=False,
            reduce=False
        )

    def train(self, iters):
        for i in range(iters):
            self.env.reset()

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            loss = self.teacher_forcing(train=True)

            # if (i == iters - 1):
            #     print("training loss: %.4f" % loss.cpu().item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 40.)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

    def visualize(self, env_name='hhh'):
        vis = []
        for _ in range(1):
            self.env.reset()
            obs = self.env._get_obs()
            obj_weight, act_weight = self.teacher_forcing(train=False, for_vis=True)
            for i, ob in enumerate(obs):
                vis.append({})                
                vis[-1]['instructions'] = self.tok.tokenize(ob['instructions'])
                vis[-1]['pred_land'] = obj_weight[i].cpu().detach()
                vis[-1]['pred_act'] = act_weight[i].cpu().detach()
                vis[-1]['scan'] = ob['scan']
                vis[-1]['gt_path'] = ob['gt_path']
        
        save = {'vis' : vis}
        save_path = os.path.join('vis', args.name, env_name + '_vis_la')
        torch.save(save, save_path)
        print('save la to:', save_path)

    def get_insts(self, wrapper=(lambda x: x)):
        # Get the caption for all the data
        self.env.reset_epoch(shuffle=True)
        path2inst = {}
        total = self.env.size()
        for _ in wrapper(range(total // self.env.batch_size + 1)):  # Guarantee that all the data are processed
            obs = self.env.reset()
            insts = self.infer_batch()  # Get the insts of the result
            path_ids = [ob['path_id'] for ob in obs]  # Gather the path ids
            for path_id, inst in zip(path_ids, insts):
                if path_id not in path2inst:
                    path2inst[path_id] = self.tok.shrink(inst)  # Shrink the words
        return path2inst

    def valid(self, *aargs, **kwargs):
        # Calculate the teacher-forcing metrics
        self.env.reset_epoch(shuffle=True)
        N = 148     # 148 x 16(batchsize) = 2368 (number of val_unseen path)
        loss = 0.
        for i in range(N):
            self.env.reset()
            loss += self.teacher_forcing(train=False)
        loss /= N

        return loss

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            state = self.env.env.sims[idx].getState()
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point) // 12   # The point idx started from 0
                trg_level = (trg_point) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                    # print("UP")
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                    # print("DOWN")
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                    # print("RIGHT")
                    # print(self.env.env.sims[idx].getState().viewIndex, trg_point)
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def _teacher_action(self, obs, ended, tracker=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def _candidate_variable(self, obs, actions):
        candidate_feat = np.zeros((len(obs), self.feature_size + args.angle_feat_size), dtype=np.float32)
        for i, (ob, act) in enumerate(zip(obs, actions)):
            if act == -1:  # Ignore or Stop --> Just use zero vector as the feature
                pass
            else:
                c = ob['candidate'][act]
                candidate_feat[i, :] = c['feature'] # Image feat
        return torch.from_numpy(candidate_feat).cuda()

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), 36, self.feature_size + args.angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']   # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def from_shortest_path(self, viewpoints=None, get_first_feat=False, get_la=False):
        """
        :param viewpoints: [[], [], ....(batch_size)]. Only for dropout viewpoint
        :param get_first_feat: whether output the first feat
        :return:
        """
        obs = self.env._get_obs()
        ended = np.array([False] * len(obs)) # Indices match permuation of the model, not env
        length = np.zeros(len(obs), np.int64)
        img_feats = []
        can_feats = []
        first_feat = np.zeros((len(obs), self.feature_size+args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            first_feat[i, -args.angle_feat_size:] = angle_feature(ob['heading'], ob['elevation'])
        first_feat = torch.from_numpy(first_feat).cuda()

        all_landmark, all_action = [], []
        all_land_mask, all_act_mask = [], []

        while not ended.all():
            if get_la:
                land_mask = torch.zeros(len(obs), dtype=torch.long)
                act_mask = torch.zeros(len(obs), dtype=torch.long)
                landmark = torch.zeros(len(obs), args.max_instr_len)
                action = torch.zeros(len(obs), args.max_instr_len)
                for i, ob in enumerate(obs):
                    land_id = torch.LongTensor(ob['landmark']).cuda() + 1  # +1 for [CLS] token
                    landmark[i][land_id] = 1
                    act_id = torch.LongTensor(ob['action']).cuda() + 1
                    action[i][act_id] = 1
                    if land_id[0] > 0:
                        land_mask[i] = 1
                    if act_id[0] > 0:
                        act_mask[i] = 1
                all_land_mask.append(land_mask)
                all_act_mask.append(act_mask)
                all_landmark.append(landmark)
                all_action.append(action)

            if viewpoints is not None:
                for i, ob in enumerate(obs):
                    viewpoints[i].append(ob['viewpoint'])
            img_feats.append(self._feature_variable(obs))
            teacher_action = self._teacher_action(obs, ended)
            teacher_action = teacher_action.cpu().numpy()
            for i, act in enumerate(teacher_action):
                if act < 0 or act == len(obs[i]['candidate']):  # Ignore or Stop
                    teacher_action[i] = -1                      # Stop Action
            can_feats.append(self._candidate_variable(obs, teacher_action))
            self.make_equiv_action(teacher_action, obs)
            length += (1 - ended)
            ended[:] = np.logical_or(ended, (teacher_action == -1))
            obs = self.env._get_obs()
        img_feats = torch.stack(img_feats, 1).contiguous()  # batch_size, max_len, 36, 2052
        can_feats = torch.stack(can_feats, 1).contiguous()  # batch_size, max_len, 2052

        if get_la:
            all_landmark = torch.stack(all_landmark, dim=1).cuda()
            all_action = torch.stack(all_action, dim=1).cuda()
            all_land_mask = torch.stack(all_land_mask, dim=1).cuda()
            all_act_mask = torch.stack(all_act_mask, dim=1).cuda()
            return (img_feats, can_feats), length, all_landmark, all_action, all_land_mask, all_act_mask
        if get_first_feat:
            return (img_feats, can_feats, first_feat), length
        else:
            return (img_feats, can_feats), length

    def gt_words(self, obs):
        """
        See "utils.Tokenizer.encode_sentence(...)" for "instr_encoding" details
        """
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]
        seq_tensor = np.zeros((len(obs), args.max_instr_len), dtype=np.int64)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
        seq_tensor = torch.from_numpy(seq_tensor)
        return seq_tensor.long().cuda()

    def teacher_forcing(self, train=True, features=None, insts=None, for_listener=False, for_vis=False, for_analyse=False):
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        # Get Image Input & Encode
        if features is not None:
            # It is used in calulating the speaker score in beam-search
            assert insts is not None
            (img_feats, can_feats), lengths = features
            ctx = self.encoder(can_feats, img_feats, lengths)
            batch_size = len(lengths)
        else:
            obs = self.env._get_obs()
            batch_size = len(obs)
            (img_feats, can_feats), lengths, all_landmark, all_action, all_land_mask, all_act_mask = self.from_shortest_path(get_la=True)   # Image Feature (from the shortest path)
        
            repeated_angle_feature = img_feats[..., -args.angle_feat_size:].repeat(1, 1, 1, 32)
            img_feats = torch.cat([img_feats[..., :-args.angle_feat_size], repeated_angle_feature], -1)
            repeated_angle_feature = can_feats[..., -args.angle_feat_size:].repeat(1, 1, 32)
            can_feats = torch.cat([can_feats[..., :-args.angle_feat_size], repeated_angle_feature], -1)

            ctx = self.encoder(can_feats, img_feats, lengths)
        h_t = torch.zeros(2 if args.LA_bidir else 1, batch_size, args.rnn_dim).cuda()
        c_t = torch.zeros(2 if args.LA_bidir else 1, batch_size, args.rnn_dim).cuda()
        ctx_mask = length2mask(lengths)

        # Get Language Input
        if insts is None:
            insts = self.gt_words(obs)                                       # Language Feature

        # Decode
        obj_weight, act_weight = self.decoder(insts, ctx, ctx_mask, h_t, c_t)

        # import pdb; pdb.set_trace();
        landmark_loss = torch.mean(F.binary_cross_entropy_with_logits(obj_weight, all_landmark, reduce=False) * \
                                    all_land_mask.unsqueeze(2).float() * \
                                    (~ctx_mask).unsqueeze(2).float())

        action_loss = torch.mean(F.binary_cross_entropy_with_logits(act_weight, all_action, reduce=False) * \
                                    all_act_mask.unsqueeze(2).float() * \
                                    (~ctx_mask).unsqueeze(2).float())

        if train:
            return landmark_loss + action_loss
        elif for_vis:
            # obj_weight = F.sigmoid(obj_weight).detach() 
            # act_weight = F.sigmoid(act_weight).detach()
            return obj_weight.detach().cpu(), act_weight.detach().cpu()
        elif for_analyse:
            return obj_weight.detach().cpu(), act_weight.detach().cpu(), all_landmark, all_action, all_land_mask, all_act_mask, ~ctx_mask
        else:
            return (landmark_loss + action_loss).cpu().item()

    def infer_batch(self, sampling=False, train=False, insts=None, featdropmask=None, features=None, for_il=False):
        """

        :param sampling: if not, use argmax. else use softmax_multinomial
        :param train: Whether in the train mode
        :return: if sampling: return insts(np, [batch, max_len]),
                                     log_probs(torch, requires_grad, [batch,max_len])
                                     hiddens(torch, requires_grad, [batch, max_len, dim})
                      And if train: the log_probs and hiddens are detached
                 if not sampling: returns insts(np, [batch, max_len])
        """
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval() 

        # Image Input for the Encoder
        # Get Image Input & Encode
        if features is not None:
            # It is used in calulating the speaker score in beam-search
            assert insts is not None
            (img_feats, can_feats), lengths = features
           
            repeated_angle_feature = img_feats[..., -args.angle_feat_size:].repeat(1, 1, 1, 32)
            img_feats = torch.cat([img_feats[..., :-args.angle_feat_size], repeated_angle_feature], -1)
            repeated_angle_feature = can_feats[..., -args.angle_feat_size:].repeat(1, 1, 32)
            can_feats = torch.cat([can_feats[..., :-args.angle_feat_size], repeated_angle_feature], -1)

            ctx = self.encoder(can_feats, img_feats, lengths, already_dropfeat=True)
            batch_size = len(lengths)
        else:
            obs = self.env._get_obs()
            batch_size = len(obs)
            viewpoints_list = [list() for _ in range(batch_size)]

            # Get feature
            (img_feats, can_feats), lengths = self.from_shortest_path(viewpoints=viewpoints_list)      # Image Feature (from the shortest path)

            # This code block is only used for the featdrop.
            # if featdropmask is not None:
            #     img_feats[..., :-args.angle_feat_size] *= featdropmask
            #     can_feats[..., :-args.angle_feat_size] *= featdropmask

            # Encoder
            ctx = self.encoder(can_feats, img_feats, lengths,
                            already_dropfeat=(featdropmask is not None))

        ctx_mask = length2mask(lengths)

        h_t = torch.zeros(2 if args.LA_bidir else 1, batch_size, args.rnn_dim).cuda()
        c_t = torch.zeros(2 if args.LA_bidir else 1, batch_size, args.rnn_dim).cuda()

        # Get Language Input
        if insts is None:
            insts = self.gt_words(obs)

        # Decode
        obj_weight, act_weight = self.decoder(insts, ctx, ctx_mask, h_t, c_t)

        hard_obj = (F.sigmoid(obj_weight) > args.hard_thd).detach().long()
        hard_act = (F.sigmoid(act_weight) > args.hard_thd).detach().long()

        if args.hard_la:
            if args.hard_mod == 'sigmoid':
                obj_weight = (F.sigmoid(obj_weight) > args.hard_thd).detach().float()
                act_weight = (F.sigmoid(act_weight) > args.hard_thd).detach().float()
            elif args.hard_mod == 'softmax':
                obj_weight = (F.softmax(obj_weight, dim=2) > args.hard_thd).detach().float()
                act_weight = (F.softmax(act_weight, dim=2) > args.hard_thd).detach().float()
            else:
                assert False, 'check hard mod'
            # import pdb; pdb.set_trace();
        else:
            obj_weight = F.softmax(obj_weight, dim=2).detach()
            act_weight = F.softmax(act_weight, dim=2).detach()

        return obj_weight, act_weight
        
    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        print("Load the laspeaker's state dict from %s" % path)
        states = torch.load(path, map_location=lambda storage, loc: storage.cuda())
        def recover_state(name, model, optimizer):
            # print(name)
            # print(list(model.state_dict().keys()))
            # for key in list(model.state_dict().keys()):
            #     print(key, model.state_dict()[key].size())
            state = model.state_dict()
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            # if args.loadOptim:
            #     optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1


