model = GIN(259, 128, 64, 0.3, 3)

train_accu = []
test_accu = []

train_loss = []
test_loss = []

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
for epoch in range(200):
    i = 0
    for batched_graph, labels in train_dataloader:  
        pred = model(batched_graph, batched_graph.ndata['f1'][:,0:259].float(),
                     batched_graph.edata['w'].float())
        loss = F.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_one_accu, train_one_loss = get_accu(train_dataloader, model, [0,259])
    test_one_accu, test_one_loss = get_accu(test_dataloader, model, [0,259])
    print(train_one_accu, test_one_accu)
    train_accu.append(train_one_accu)
    test_accu.append(test_one_accu)
    train_loss.append(train_one_loss)
    test_loss.append(test_one_loss)
    
