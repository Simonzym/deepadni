#gnn training (progress visible)
train_sampler = SubsetRandomSampler(torch.arange(400))
test_sampler = SubsetRandomSampler(torch.arange(400))

train_dataloader = GraphDataLoader(train_graph, batch_size = 20, 
                                   sampler = train_sampler,  drop_last = False)
test_dataloader = GraphDataLoader(test_graph, batch_size = 20, 
                                  sampler = test_sampler,  drop_last = False)

model = GIN(128, 128,  64, 0, 2)

train_accu = []
test_accu = []

train_loss = []
test_loss = []

auc_pred = []
auc_label = []
#metrics.roc_auc_score(auc_label, auc_pred)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
for epoch in range(100):
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata['f1'].float(),
                     batched_graph.edata['w'].float())
        loss = F.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_one_accu, train_one_loss,_ = get_accu(train_dataloader, model)
    test_one_accu, test_one_loss,_ = get_accu(test_dataloader, model)
    print(train_one_accu, test_one_accu)
    train_accu.append(train_one_accu)
    test_accu.append(test_one_accu)
    train_loss.append(train_one_loss)
    test_loss.append(test_one_loss)
    
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)