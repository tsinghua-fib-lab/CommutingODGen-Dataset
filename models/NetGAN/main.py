import time

from data_load import *
from metrics import *
from model import *

from pprint import pprint

print("\n  **Loading data...")
nfeats_train, adjs_train, dises_train, ods_train, nfeats_valid, adjs_valid, dises_valid, ods_valid, nfeats_test, adjs_test, dises_test, ods_test = load_data()

nfeat_scaler, dis_scaler, od_scaler = get_scalers(nfeats_train, dises_train, ods_train)

generator = Generator()
discriminator = Discriminator()

generator = generator.cuda()
discriminator = discriminator.cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=3e-4)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=3e-4)


print('\n  **Start training...')
start = time.time()

for epoch in range(2):
    start_epoch = time.time()
    print(f"Epoch {epoch+1}:", end=" | ")
    generator.train()
    discriminator.train()

    loss_G_epoch = []
    loss_D_epoch = []
    for nfeat, adj, dis, od in zip(nfeats_train, adjs_train, dises_train, ods_train):
        optimizer_G.zero_grad()

        nfeat = nfeat_scaler.transform(nfeat)
        dis = dis_scaler.transform(dis.reshape(-1, 1)).reshape(dis.shape)
        od = od_scaler.transform(od.reshape(-1, 1)).reshape(od.shape)

        nfeat = torch.FloatTensor(nfeat).cuda()
        g = build_graph(adj).to(torch.device('cuda'))
        dis = torch.FloatTensor(dis).cuda()
        # od = torch.FloatTensor(od).cuda()

        fake_batch = generator.sample_generated_batch(g, nfeat, dis, 128).to(torch.device('cuda'))

        loss_G = -torch.mean(discriminator(fake_batch))
        loss_G.backward()
        optimizer_G.step()
        loss_G_value = loss_G.item()
        loss_G_epoch.append(loss_G_value)
    

        if epoch % 5 == 0:
            optimizer_D.zero_grad()

            with torch.no_grad():
                _, adjacency, logp = generator.generate_OD_net(g, nfeat, dis)
                batch = []
                for _ in range(128):
                    one_seq = sample_one_random_walk(adjacency, logp)
                    batch.append(one_seq)
                batch = torch.stack(batch).to(torch.device('cuda'))
                fake_batch = batch

            real_batch = sample_batch_real(od)
            real_batch = torch.FloatTensor(real_batch).to(torch.device('cuda'))
        
            loss_D_fake = torch.mean(discriminator(fake_batch))
            loss_D_real = -torch.mean(discriminator(real_batch))

            loss_D = loss_D_fake + loss_D_real + 10 * compute_gradient_penalty(discriminator, real_batch, fake_batch)
            loss_D_value = loss_D.item()
            loss_D.backward()
            optimizer_D.step()

            loss_D_epoch.append(loss_D_value)

    loss_G_epoch = np.mean(loss_G_epoch)
    if epoch % 5 == 0:
        loss_D_epoch = np.mean(loss_D_epoch)

    print(f"loss_G: {loss_G_epoch:.4f}", end=" | ")
    if epoch % 5 == 0:
        print(f"loss_D: {loss_D_epoch:.4f}", end=" | ")
    
    print(f"Time: {time.time()-start_epoch:.2f}s")


print('Complete!', end=" ")
print('Consume ', time.time()-start, ' seconds!')
print("-"*50)

print("\n  **Evaluating...")
generator.eval()
metrics_all = []
for nfeat, adj, dis, od in zip(nfeats_test, adjs_test, dises_test, ods_test):
    nfeat = nfeat_scaler.transform(nfeat)
    dis = dis_scaler.transform(dis.reshape(-1, 1)).reshape(dis.shape)
    nfeat = torch.FloatTensor(nfeat).cuda()
    dis = torch.FloatTensor(dis).cuda()
    g = build_graph(adj).to(torch.device('cuda'))

    with torch.no_grad():
        OD_gen, _, _ = generator.generate_OD_net(g, nfeat, dis)
        OD_gen = OD_gen.cpu().numpy()
        OD_gen = od_scaler.inverse_transform(OD_gen)
        OD_gen[OD_gen < 0] = 0

        metrics = cal_od_metrics(OD_gen, od)
        metrics_all.append(metrics)

avg_metrics = average_listed_metrics(metrics_all)
pprint(avg_metrics)