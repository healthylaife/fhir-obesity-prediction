import importlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import *

import parameters

importlib.reload(parameters)
from parameters import *


class EncDec2(nn.Module):
    def __init__(self, device, feat_vocab_size, age_vocab_size, demo_vocab_size, embed_size, rnn_size, batch_size):
        super(EncDec2, self).__init__()
        self.embed_size = embed_size
        self.latent_size = args.latent_size
        self.rnn_size = rnn_size
        self.feat_vocab_size = feat_vocab_size

        self.age_vocab_size = age_vocab_size

        self.demo_vocab_size = demo_vocab_size
        self.batch_size = batch_size
        self.padding_idx = 0
        self.device = device
        self.build()

    def build(self):

        self.emb_feat = FeatEmbed(self.device, self.feat_vocab_size, self.embed_size, self.batch_size)

        self.emb_age = AgeEmbed(self.device, self.age_vocab_size, self.embed_size, self.batch_size)

        self.emb_demo = DemoEmbed(self.device, self.demo_vocab_size, self.embed_size, self.batch_size)

        self.enc = Encoder2(self.device, self.feat_vocab_size, self.age_vocab_size, self.embed_size, self.rnn_size,
                            self.batch_size)
        self.dec = Decoder2(self.device, self.feat_vocab_size, self.age_vocab_size, self.embed_size, self.rnn_size,
                            self.emb_feat, self.batch_size)

    def forward(self, visualize_embed, find_contri, enc_feat, enc_len, enc_age, enc_demo, dec_feat):
        if visualize_embed:
            features = torch.tensor(np.arange(0, 350))
            features = features.type(torch.LongTensor)
            features = features.to(self.device)
            emb = self.emb_feat(features)
            return emb

        contri = torch.cat((enc_feat.unsqueeze(2), enc_age.unsqueeze(2)), 2)

        enc_feat = self.emb_feat(enc_feat)
        enc_age = self.emb_age(enc_age)
        enc_demo = self.emb_demo(enc_demo)

        code_output, code_h_n, code_c_n = self.enc(enc_feat, enc_len, enc_age)

        dec_feat = self.emb_feat(dec_feat)
        dec_feat = torch.sum(dec_feat, 2)


        if find_contri:
            dec_output, dec_prob, disc_input, kl_input, all_contri = self.dec(find_contri, contri, dec_feat, code_output, code_h_n, enc_demo)
        else:
            dec_output, dec_prob, disc_input, kl_input = self.dec(find_contri, contri, dec_feat, code_output, code_h_n, enc_demo)
        kl_input = torch.tensor(kl_input)

        disc_input = torch.stack(disc_input)

        disc_input = torch.reshape(disc_input, (args.time, -1, disc_input.shape[1]))

        disc_input = disc_input.permute(1, 0, 2)

        kl_input = torch.reshape(kl_input, (args.time, -1))
        kl_input = kl_input.permute(1, 0)

        if find_contri:
            return dec_output, dec_prob, kl_input, all_contri
        else:
            return dec_output, dec_prob, kl_input, disc_input


class Encoder2(nn.Module):
    def __init__(self, device, feat_vocab_size, age_vocab_size, embed_size, rnn_size, batch_size):
        super(Encoder2, self).__init__()
        self.embed_size = embed_size
        self.rnn_size = rnn_size
        self.feat_vocab_size = feat_vocab_size

        self.age_vocab_size = age_vocab_size

        self.padding_idx = 0
        self.device = device
        self.build()

    def build(self):
        self.rnn = nn.LSTM(input_size=self.embed_size * 2, hidden_size=self.rnn_size, num_layers=args.rnnLayers,
                           batch_first=True)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, featEmbed, lengths, ageEmbed):
        out1 = torch.cat((featEmbed, ageEmbed), 2)

        out1 = out1.type(torch.FloatTensor)
        out1 = out1.to(self.device)

        h_0, c_0 = self.init_hidden(featEmbed.shape[0])
        h_0, c_0 = h_0.to(self.device), c_0.to(self.device)

        lengths = lengths.type(torch.LongTensor)

        code_pack = torch.nn.utils.rnn.pack_padded_sequence(out1, lengths, batch_first=True, enforce_sorted=False)

        code_output, (code_h_n, code_c_n) = self.rnn(code_pack, (h_0, c_0))
        code_h_n = code_h_n[-1, :, :].squeeze()
        code_c_n = code_c_n[-1, :, :].squeeze()
        code_output, _ = torch.nn.utils.rnn.pad_packed_sequence(code_output, batch_first=True)

        return code_output, code_h_n, code_c_n

    def init_hidden(self, batch_size):
        h = torch.zeros(args.rnnLayers, batch_size, self.rnn_size)
        c = torch.zeros(args.rnnLayers, batch_size, self.rnn_size)

        h = Variable(h)
        c = Variable(c)

        return (h, c)


class Decoder2(nn.Module):
    def __init__(self, device, feat_vocab_size, age_vocab_size, embed_size, rnn_size, emb_feat, batch_size):
        super(Decoder2, self).__init__()
        self.embed_size = embed_size

        self.rnn_size = rnn_size
        self.feat_vocab_size = feat_vocab_size

        self.age_vocab_size = age_vocab_size

        self.padding_idx = 0
        self.device = device
        self.emb_feat = emb_feat
        self.build()

    def build(self):

        self.linears = nn.ModuleList([nn.Linear(
            (2 * self.rnn_size) + 5 * int(self.embed_size / 2) + 4 * int(self.embed_size), 2 * (args.latent_size)) for i
            in range(args.time)])
        self.linearsMed = nn.ModuleList([nn.Linear(2 * (args.latent_size), args.latent_size) for i in range(args.time)])
        self.linearsLast = nn.ModuleList([nn.Linear(args.latent_size, 1) for i in range(args.time)])

        self.leaky = nn.ModuleList([nn.LeakyReLU(0.1) for i in range(args.time)])

        self.drop = nn.Dropout(p=0.2)

        self.attn = LSTMAttention(self.rnn_size)

    def forward(self, find_contri, contri, featEmbed, encoder_outputs, h_n, enc_demo):
        dec_output = []
        kl_input = []
        disc_input = []
        dec_prob = []
        for t in range(args.time):

            a = self.attn(encoder_outputs)
            if (find_contri) and (t == 2):
                all_contri = self.attention(contri, a)
            a = a.unsqueeze(1)

            weighted = torch.bmm(a, encoder_outputs)

            weighted = weighted.permute(1, 0, 2)
            weighted = weighted.squeeze()

            reg_input = torch.cat((h_n, weighted), dim=1)
            reg_input = torch.cat((reg_input, enc_demo), dim=1)

            for i in range(featEmbed.shape[1]):
                reg_input = torch.cat((reg_input, featEmbed[:, i, :]), dim=1)

            bmi_h = self.linears[t](reg_input)
            bmi_h = self.leaky[t](bmi_h)
            if hasattr(self, "linearsMed"):
                bmi_h = self.linearsMed[t](bmi_h)
            bmi_h = self.linearsLast[t](bmi_h)
            bmi_prob = torch.sigmoid(bmi_h)
            bmi_prob_non = (1 - bmi_prob[:, 0]).unsqueeze(1)
            bmi_prob = torch.cat((bmi_prob_non, bmi_prob), axis=1)

            bmi_label = torch.argmax(bmi_prob, dim=1)
            dec_output.append(bmi_label)

            d = {0: 0, 1: 1}
            bmi_label_dict = torch.tensor([d[x.item()] for x in bmi_label])

            kl_input.extend(bmi_label_dict)
            disc_input.extend(bmi_h)
            dec_prob.append(bmi_prob)

        if find_contri:
            return dec_output, dec_prob, disc_input, kl_input, all_contri
        else:
            return dec_output, dec_prob, disc_input, kl_input

    def attention(self, contri, attn):

        contri = contri[:attn.shape[0], :attn.shape[1], :]
        attn = attn.unsqueeze(2)
        attn = attn.to('cpu')
        contri = contri.type(torch.FloatTensor)
        contri = torch.cat((contri, attn), 2)
        contri = contri.type(torch.FloatTensor)
        return contri


class LSTMAttention(nn.Module):
    def __init__(self, rnn_size):
        super().__init__()
        self.attn = nn.Linear((rnn_size), int(rnn_size / 2))
        self.v = nn.Linear(int(rnn_size / 2), 1, bias=False)

    def forward(self, encoder_outputs):
        energy = torch.tanh(self.attn(encoder_outputs))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class FeatEmbed(nn.Module):
    def __init__(self, device, feat_vocab_size, embed_size, batch_size):
        super(FeatEmbed, self).__init__()
        self.embed_size = embed_size
        self.feat_vocab_size = feat_vocab_size
        self.padding_idx = 0
        self.device = device
        self.build()

    def build(self):
        self.emb_feat = nn.Embedding(self.feat_vocab_size, self.embed_size, self.padding_idx)

    def forward(self, feat):
        feat = feat.type(torch.LongTensor)
        feat = feat.to(self.device)
        featEmbed = self.emb_feat(feat)
        featEmbed = featEmbed.type(torch.FloatTensor)
        featEmbed = featEmbed.to(self.device)
        return featEmbed


class AgeEmbed(nn.Module):
    def __init__(self, device, age_vocab_size, embed_size, batch_size):
        super(AgeEmbed, self).__init__()
        self.embed_size = embed_size
        self.age_vocab_size = age_vocab_size
        self.padding_idx = 0
        self.device = device
        self.build()

    def build(self):
        self.emb_age = nn.Embedding(self.age_vocab_size, self.embed_size, self.padding_idx)

    def forward(self, age):
        age = age.type(torch.LongTensor)
        age = age.to(self.device)
        ageEmbed = self.emb_age(age)
        ageEmbed = ageEmbed.type(torch.FloatTensor)
        ageEmbed = ageEmbed.to(self.device)
        return ageEmbed


class DemoEmbed(nn.Module):
    def __init__(self, device, demo_vocab_size, embed_size, batch_size):
        super(DemoEmbed, self).__init__()
        self.embed_size = embed_size
        self.demo_vocab_size = demo_vocab_size
        self.padding_idx = 0
        self.device = device
        self.build()

    def build(self):
        self.emb_demo = nn.Embedding(self.demo_vocab_size, self.embed_size, self.padding_idx)
        self.fc = nn.Linear(self.embed_size, int(self.embed_size / 2))

    def forward(self, demo):
        demo = demo.type(torch.LongTensor)
        demo = demo.to(self.device)
        demoEmbed = self.emb_demo(demo)
        demoEmbed = self.fc(demoEmbed)
        demoEmbed = torch.reshape(demoEmbed, (demoEmbed.shape[0], -1))
        demoEmbed = demoEmbed.type(torch.FloatTensor)
        demoEmbed = demoEmbed.to(self.device)
        return demoEmbed

