def print_msg(msg):

    msg = "## {} ##".format(msg)
    length = len(msg)
    msg = "\n{}\n".format(msg)
    print(length*"#" + msg + length * "#")
import torch
def cross_entropy(y,y_pre,weight):

    res = weight * (y*torch.log(y_pre))
    loss=-torch.sum(res)
    return loss/y_pre.shape[0]

    model_file = 'epochs:{}_alpha:{}_anchor:{}_sample_max_path_len:{}_embedding_size{}_learning_rate{}_batch_size{}_get_top_k{}'.format(
        args.epochs,
        args.alpha,
        args.anchor,
        args.sample_max_path_len,
        args.embedding_size,
        args.learning_rate,
        args.batch_size,
        args.get_top_k
    )
    os.makedirs("../rules/{}/{}/{}".format(args.model,args.datasets,model_file), exist_ok=True)
