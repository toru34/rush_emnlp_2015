import _dynet as dy

class ABS:
    def __init__(self, emb_dim, hid_dim, vocab_size, q, c, model, encoder_type='attention'):
        # Neural language model
        self.E = model.add_lookup_parameters((vocab_size, emb_dim))
        self._U = model.add_parameters((hid_dim, emb_dim*c))
        self._V = model.add_parameters((vocab_size, hid_dim))
        self.c = c
        # Encoder
        self.F = model.add_lookup_parameters((vocab_size, hid_dim))
        self._W = model.add_parameters((vocab_size, hid_dim))
        self.encoder_type = encoder_type
        self.q = q

        if self.encoder_type == 'attention':
            # Attention-based encoder
            self.G = model.add_lookup_parameters((vocab_size, emb_dim))
            self._P = model.add_parameters((hid_dim, emb_dim*c))

    def __call__(self, x, t):
        # Neural language model
        x_embs = [dy.lookup(self.F, x_t) for x_t in x]
        t_embs = [dy.lookup(self.E, t_t) for t_t in t]

        y = []
        if self.encoder_type == 'bow':
            enc = dy.average(x_embs)
            W_enc = self.W*enc
            for i in range(len(t)-self.c+1):
                # Neural language model
                t_c = dy.concatenate(t_embs[i:i+self.c])
                h = dy.tanh(self.U*t_c)

                # Output layer
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

    def associate_parameters(self):
        self.U = dy.parameter(self._U)
        self.V = dy.parameter(self._V)
        self.W = dy.parameter(self._W)
        if self.encoder_type == 'attention':
            self.P = dy.parameter(self._P)
