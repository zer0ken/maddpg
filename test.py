import torch as T

T.set_default_dtype(T.float32)

a = T.tensor(1., requires_grad=False)
b = T.tensor(1., requires_grad=True)
b2 = T.tensor(1., requires_grad=True)

bop = T.optim.RMSprop([b, b2], lr=0.1)

for i in range(10):
    print('===', i)
        
    c = a * b
    d = c ** 2 - b2

    print('pred:', d)


    result = T.tensor(20, requires_grad=False)
    loss = result - d

    print('loss:', loss)

    bop.zero_grad()
    loss.backward(gradient=loss)
    bop.step()

    print('b:', b)
    print('b2:', b2)