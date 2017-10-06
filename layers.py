import _dynet as dy

class ABS:
    def __init__(self, model, emb_dim, hid_dim, vocab_size, q, c, encoder_type='attention'):
        pc = model.add_subcollection()

        # Neural language model
        self.E  = pc.add_lookup_parameters((vocab_size, emb_dim))
        self._U = pc.add_parameters((hid_dim, c*emb_dim))
        self._V = pc.add_parameters((vocab_size, hid_dim))
        self.c  = c

        # Encoder
        self.F            = pc.add_lookup_parameters((vocab_size, hid_dim))
        self._W           = pc.add_parameters((vocab_size, hid_dim))
        self.encoder_type = encoder_type
        self.q            = q

        # Attention-based encoder
        if self.encoder_type == 'attention':
            self.G  = pc.add_lookup_parameters((vocab_size, emb_dim))
            self._P = pc.add_parameters((hid_dim, c*emb_dim))

        self.pc = pc
        self.spec = (emb_dim, hid_dim, vocab_size, q, c, encoder_type)

    def __call__(self, x=None, t=None, test=False):
        if test:
            tt_embs = [dy.lookup(self.E, t_t) for t_t in t]

            if self.encoder_type == 'bow':
                # Neural language model
                tt_c    = dy.concatenate(tt_embs)
                h       = dy.tanh(self.U*tt_c)

                # Output with softmax
                y_t     = dy.softmax(self.V*h + self.W_enc)

            elif self.encoder_type == 'attention':
                ttp_embs = [dy.lookup(self.G, t_t) for t_t in t]

                # Neural language model
                tt_c = dy.concatenate(tt_embs)
                h    = dy.tanh(self.U*tt_c)

                # Attention
                ttp_c = dy.concatenate(ttp_embs)
                p     = dy.softmax(self.xt*self.P*ttp_c) # Attention weight
                enc   = self.xb*p                        # Context vector

                # Output with softmax
                y_t = dy.softmax(self.V*h + self.W*enc)

            return y_t


        else:
            xt_embs = [dy.lookup(self.F, x_t) for x_t in x]
            tt_embs = [dy.lookup(self.E, t_t) for t_t in t]

            y = []
            if self.encoder_type == 'bow':
                # BoW
                enc = dy.average(xt_embs)
                W_enc = self.W*enc
                for i in range(len(t)-self.c+1):
                    # Neural language model
                    tt_c = dy.concatenate(tt_embs[i:i+self.c])
                    h = dy.tanh(self.U*tt_c)

                    # Output without softmax
                    y_t = self.V*h + W_enc
                    y.append(y_t)

            elif self.encoder_type == 'attention':
                xb = dy.concatenate([dy.esum(xt_embs[max(i-self.q,0):min(len(x)-1+1,i+self.q+1)])/self.q for i in range(len(x))], d=1)
                xt = dy.transpose(dy.concatenate(xt_embs, d=1))
                ttp_embs = [dy.lookup(self.G, t_t) for t_t in t]

                for i in range(len(t)-self.c+1):
                    # Neural language model
                    tt_c = dy.concatenate(tt_embs[i:i+self.c])
                    h = dy.tanh(self.U*tt_c)

                    # Attention
                    ttp_c = dy.concatenate(ttp_embs[i:i+self.c]) # Window-sized embedding
                    p     = dy.softmax(xt*self.P*ttp_c) # Attention weight
                    enc = xb*p # Context vector

                    # Output without softmax
                    y_t = self.V*h + self.W*enc
                    y.append(y_t)

            return y

    def set_initial_states(self, x):
        self.xt_embs = [dy.lookup(self.F, x_t) for x_t in x]

        if self.encoder_type == 'bow':
            self.W_enc = self.W*dy.average(self.xt_embs)

        elif self.encoder_type == 'attention':
            self.xb       = dy.concatenate(
                [dy.esum(self.xt_embs[max(i-self.q,0):min(len(x)-1+1,i+self.q+1)])/self.q for i in range(len(x))],
                d=1
            )
            self.xt       = dy.transpose(dy.concatenate(self.xt_embs, d=1))

    def associate_parameters(self):
        self.U = dy.parameter(self._U)
        self.V = dy.parameter(self._V)
        self.W = dy.parameter(self._W)

        if self.encoder_type == 'attention':
            self.P = dy.parameter(self._P)

    @staticmethod
    def from_spec(spec, model):
        emb_dim, hid_dim, vocab_size, q, c, encoder_type = spec
        return ABS(model, emb_dim, hid_dim, vocab_size, q, c, encoder_type)

    def param_collection(self):
        return self.pc
