import os
if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
    import _dynet as dy  # Use cpu
else:
    import _gdynet as dy # Use gpu

class ABS:
    def __init__(self, model, emb_dim, hid_dim, vocab_size, q, c, encoder_type='attention'):
        pc = model.add_subcollection()

        # Neural language model
        self.E = pc.add_lookup_parameters((vocab_size, emb_dim))
        self._U = pc.add_parameters((hid_dim, c*emb_dim))
        self._V = pc.add_parameters((vocab_size, hid_dim))
        self.c = c

        # Encoder
        self.F = pc.add_lookup_parameters((vocab_size, hid_dim))
        self._W = pc.add_parameters((vocab_size, hid_dim))
        self.encoder_type = encoder_type
        self.q = q

        # Attention-based encoder
        if self.encoder_type == 'attention':
            self.G = pc.add_lookup_parameters((vocab_size, emb_dim))
            self._P = pc.add_parameters((hid_dim, c*emb_dim))

        self.pc = pc
        self.spec = (emb_dim, hid_dim, vocab_size, q, c, encoder_type)

    def __call__(self, x=None, t=None, tm1s=None, test=False):
        if test:
            self.t_embs = self.t_embs[1:] + [dy.lookup(self.E, t)]

            # Neural language model
            t_c = dy.concatenate(self.t_embs)
            h = dy.tanh(self.U*t_c)

            if self.encoder_type == 'bow':
                # Outpu layer
                y_t = dy.softmax(self.V*h + self.W*self.bow_enc)

                return y_t
            elif self.encoder_type == 'attention':
                self.tp_embs = self.tp_embs[1:] + [dy.lookup(self.G, t)]

                # Attention-based encoder
                tp_c = dy.concatenate(self.tp_embs)
                p = dy.softmax(self.x_embs*self.P*tp_c) # Attention weight
                enc = self.xb_embs*p

                # Output layer
                y_t = dy.softmax(self.V*h + self.W*enc)

                return y_t
        else:
            x_embs = [dy.lookup(self.F, x_t) for x_t in x]
            t_embs = [dy.lookup(self.E, t_t) for t_t in t]

            y = []
            if self.encoder_type == 'bow':
                enc = dy.average(x_embs)
                W_enc = self.W*enc
                for i in range(len(t)-self.c+1):
                    t_c = dy.concatenate(t_embs[i:i+self.c])
                    h = dy.tanh(self.U*t_c)

                    # Output
                    y_t = self.V*h + W_enc
                    y.append(y_t)

            elif self.encoder_type == 'attention':
                x_embs = dy.transpose(dy.concatenate(x_embs, d=1))
                xb_embs = dy.concatenate([dy.mean_dim(x_embs[max(i-self.q,0):min(len(x)-1,i+self.q)],d=0) for i in range(len(x))], d=1)
                tp_embs = [dy.lookup(self.G, t_t) for t_t in t]

                for i in range(len(t)-self.c+1):
                    # Neural language model
                    t_c = dy.concatenate(t_embs[i:i+self.c])
                    h = dy.tanh(self.U*t_c)

                    # Attention-based encoder
                    tp_c = dy.concatenate(tp_embs[i:i+self.c])
                    p = dy.softmax(x_embs*self.P*tp_c) # Attention weight
                    enc = xb_embs*p

                    # Output layer
                    y_t = self.V*h + self.W*enc
                    y.append(y_t)

            return y

    def set_init_states(self, x, t):
        self.x_embs = dy.transpose(dy.concatenate([dy.lookup(self.F, x_t) for x_t in x], d=1))
        self.xb_embs = dy.concatenate([dy.mean_dim(self.x_embs[max(i-self.q,0):min(len(x)-1,i+self.q)],d=0) for i in range(len(x))], d=1)
        self.t_embs = [dy.lookup(self.E, t_t) for t_t in t] # 4つ
        if self.encoder_type == 'bow':
            self.bow_enc = dy.mean_dim(self.x_embs, d=0)
        elif self.encoder_type == 'attention':
            self.tp_embs = [dy.lookup(self.G, t_t) for t_t in t] # 4つ

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
