import torch

def one_hot(index_list,class_num):
    '''
    可以将索引变为one_hot形式的张量
    :param index_list: [2,1,3,0]
    :param class_num: 类别数量
    :return:
    tensor([[0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.]])
    '''
    if type(index_list) == torch.Tensor:
        index_list = index_list.detach().numpy()
    indexes = torch.LongTensor(index_list).view(-1,1)
    out = torch.zeros(len(index_list),class_num)
    # scatter_(input, dim, index, src)：将src中数据根据index中的索引按照dim的方向填进input。可以理解成放置元素或者修改元素     
    out = out.scatter_(dim=1,index=indexes,value=1)
    return out

if __name__ == '__main__':
    a=[1,0,2,1]
    b=[[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
    b = torch.Tensor(b)
    a = one_hot(a,3)
    print(a)
    print(b)
    print(a*b)
    # 不加.sum直接a*b会变成n*m矩阵,axis=0/1为按列/行相加
    # [2.,1.,3.] 就是action向量
    print( (a * b).sum(1))

