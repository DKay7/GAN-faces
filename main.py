from net import NetworkStuff

ns = NetworkStuff(file_name='gan_1', num_epochs=7)

ns.train()
ns.plotter()
ns.make_gif()
