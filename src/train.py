import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    model.train()
    if not os.path.isdir(args.snapshot_save_dir): os.makedirs(args.snapshot_save_dir)
    with open(os.path.join(args.snapshot_save_dir, 'train.csv'), 'w') as trainF, \
            open(os.path.join(args.snapshot_save_dir, 'test.csv'), 'w') as testF:
        for epoch in range(1, args.epochs+1):
            for batch in train_iter:
                feature, target = batch.text, batch.label
                feature.data.t_(), target.data.sub_(1)  # batch first, index align
                if args.cuda:
                    feature, target = feature.cuda(), target.cuda()

                optimizer.zero_grad()
                logit = model(feature)
                loss = F.cross_entropy(logit, target)
                loss.backward()
                optimizer.step()

                steps += 1
                if steps % args.log_interval == 0:
                    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                    error = (batch.batch_size - corrects)/batch.batch_size * 100.0

                    sys.stdout.write(
                        '\rBatch[{}] - loss: {:.6f}  error: {:.4f}%({}/{})'.format(steps,
                                                                                 loss.data[0],
                                                                                 error,
                                                                                 corrects,
                                                                                 batch.batch_size))
                    trainF.write('{},{},{},{},{}\n'.format(steps, loss.data[0], error, corrects, batch.batch_size))
                    trainF.flush()

                if steps % args.test_interval == 0:
                    eval_avg_loss, eval_error, eval_corrects, eval_size = eval(dev_iter, model, args)
                    testF.write('{},{},{},{},{}\n'.format(steps, eval_avg_loss, eval_error, eval_corrects, eval_size))
                    testF.flush()

                if steps % args.save_interval == 0:
                    save_prefix = os.path.join(args.snapshot_save_dir, 'snapshot')
                    save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                    torch.save(model, save_path)


def eval(data_iter, model, args):

    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    error = (size - corrects)/size * 100.0
    model.train()
    print('\nEvaluation - loss: {:.6f}  error: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       error,
                                                                       corrects,
                                                                       size))
    return avg_loss, error, corrects, size

def predict(text, model, text_field, label_feild):
    assert isinstance(text, str)
    model.eval()
    text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0][0]+1]