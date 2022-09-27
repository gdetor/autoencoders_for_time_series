import matplotlib.pylab as plt
from torch import load, randn
from torch.autograd import Variable
from vae_analyzer import VAE, encoder, decoder, GELU


if __name__ == '__main__':
    net = load("./data/vae_model_indicators.pt")
    print(net)

    w1 = net.decoder.net_decoder.weight.data.detach().cpu().numpy()
    w2 = net.encoder.net_encoder.weight.data.detach().cpu().numpy()
    ws = net.s.weight.data.detach().cpu().numpy()
    wm = net.m.weight.data.detach().cpu().numpy()

    plt.figure()
    for _ in range(3):
        sample = Variable(randn(1, 32)).cuda()
        res = net.decoder(sample).detach().cpu().numpy()
        plt.plot(res[0, :])
    # plt.plot(w1[:, 3])

    plt.figure()
    plt.subplot(221)
    # plt.plot(ws)
    plt.imshow(ws, aspect='auto')
    plt.subplot(222)
    # plt.plot(wm)
    plt.imshow(wm, aspect='auto')
    plt.subplot(223)
    # plt.plot(w1)
    plt.imshow(w1, aspect='auto')
    plt.subplot(224)
    # plt.plot(w2)
    plt.imshow(w2, aspect='auto')

    plt.show()
