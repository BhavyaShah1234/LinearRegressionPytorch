import torch

# TWEAK DATASET
rows = int(input('ENTER NUMBER OF ROWS IN DATASET: '))
cols_y = int(input('ENTER NUMBER OF TARGET COLUMNS: '))
cols_x = int(input('ENTER NUMBER OF INPUT COLUMNS: '))
min_loss = 0.03
max_iter = 10000

# INITIALIZATIONS
counter = 0
l = torch.inf
y = torch.rand(rows, cols_y)
x = torch.rand(rows, cols_x)
w0 = torch.rand(cols_y, cols_x, requires_grad=True)
b0 = torch.rand(cols_y, requires_grad=True)


# LINEAR MODEL
def model(inp, true, w, b, lr):
    w.requires_grad = True
    b.requires_grad = True
    pred = torch.add(torch.matmul(inp, w.t()), b)
    diff = torch.subtract(pred, true)
    loss = torch.divide(torch.sum(torch.square(diff)), diff.numel())
    loss.backward()
    gw = w.grad
    gb = b.grad
    with torch.no_grad():
        w = torch.subtract(w, torch.multiply(lr, gw))
        b = torch.subtract(b, torch.multiply(lr, gb))
        gw.zero_()
        gb.zero_()
    return w, b, loss


# TRAIN MODEL
while l > min_loss and counter < max_iter:
    w1, b1, l = model(x, y, w0, b0, 0.1)
    print(f'LOSS: {l}, COUNTER: {counter + 1}')
    # print(w1)
    # print(b1, end='\n\n')
    counter = counter + 1
    w0, b0 = w1, b1
