from net import NetworkStuff

ns = NetworkStuff(file_name='gan_1', num_epochs=7)

# ns.train()
# ns.plotter()
ns.load_data()
ns.load_model()
# ns.make_gif()
ns.show_samples()
