from net import NetworkStuff

ns = NetworkStuff('gan_1')
# ns.train()
ns.load_data()
ns.load_model()
ns.show_samples()
# ns.plotter()
# ns.make_gif()
