train1_ex = pd.read_csv('Code/Info/SimGraph/graphSim1/train/nodes.csv')
test1_ex = pd.read_csv('Code/Info/SimGraph/graphSim1/test/nodes.csv')

train2_ex = pd.read_csv('Code/Info/SimGraph/graphSim2/train/nodes.csv')
test2_ex = pd.read_csv('Code/Info/SimGraph/graphSim2/test/nodes.csv')

def convert_seq(folder, set_type = 'train'):
    
    path = ''.join(['Code/Info/SimGraph/', folder, '/', set_type, '/nodes.csv'])
    dataset = pd.read_csv(path)
    all_gid = list(set(dataset['graph_id']))
    seq = []
    for gid in all_gid:
        
        image_gid = np.array(dataset.loc[dataset['graph_id'] == gid])
        image_gid = np.delete(image_gid, -1, axis=1)
        num_img = image_gid.shape[0]
        if num_img < 9:
            sup_img = np.zeros((9 - num_img, 128)) - 1
            image_gid = np.vstack([image_gid, sup_img])
            
        seq.append(list(image_gid))
    return np.array(seq)

train1_seq = convert_seq(train1_ex)
train2_seq = convert_seq(train2_ex)
test1_seq = convert_seq(test1_ex)
test2_seq = convert_seq(test2_ex)