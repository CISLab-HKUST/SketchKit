import numpy as np
from sklearn.metrics import average_precision_score


def accuracy_score(true_class, pred_class):
    true_class_dshape = len(true_class.shape)
    pred_class_dshape = len(pred_class.shape)
    assert true_class_dshape == 1

    if true_class_dshape == pred_class_dshape:
        score = true_class == pred_class  # (B)
    else:
        assert pred_class_dshape == 2
        k = pred_class.shape[1]
        true_class_tile = np.tile(np.expand_dims(true_class, axis=-1), [1, k])

        score = true_class_tile == pred_class  # (B, k)
        score = np.any(score, axis=-1)  # (B)

    accuracy = np.mean(score)
    return accuracy


def mAP_score(true_class, pred_probs, n_classes):
    # print('### true_class', true_class.shape)
    # print('### pred_probs', pred_probs.shape)
    # pred_class = np.argmax(pred_probs, axis=1)
    # print('### pred_class', pred_class.shape)
    # top_1_acc = np.sum(true_class == pred_class) / float(len(true_class))
    # print('### top_1_acc', top_1_acc * 100)
    #
    # pred_sorted_labels = np.argsort(pred_probs, axis=-1)
    # pred_sorted_labels_top3 = pred_sorted_labels[:, -3:]  # (B, 3)
    # pred_sorted_labels_top5 = pred_sorted_labels[:, -5:]  # (B, 5)
    # pred_sorted_labels_top10 = pred_sorted_labels[:, -10:]  # (B, 10)
    #
    # eval_acc_top3 = accuracy_score(true_class, pred_sorted_labels_top3)
    # eval_acc_top5 = accuracy_score(true_class, pred_sorted_labels_top5)
    # eval_acc_top10 = accuracy_score(true_class, pred_sorted_labels_top10)
    # print('### top_3_acc', eval_acc_top3 * 100)
    # print('### top_5_acc', eval_acc_top5 * 100)
    # print('### top_10_acc', eval_acc_top10 * 100)

    ap_list = []
    for i in range(n_classes):
        y_true = true_class == i
        y_score = pred_probs[:, i]
        ap = average_precision_score(y_true, y_score)
        ap_list.append(ap)
    mAP = np.mean(ap_list)
    return mAP


def recognition_metrics(gt_labels, predict_probs, class_number):
    """

    Args:
        gt_labels: (batch_size)
        predict_probs: (batch_size, num_class), predicted probability
        class_number:

    Returns:

    """
    pred_sorted_labels = np.argsort(predict_probs, axis=-1)
    pred_sorted_labels_top1 = np.expand_dims(pred_sorted_labels[:, -1], axis=-1)  # (B, 1)
    pred_sorted_labels_top3 = pred_sorted_labels[:, -3:]  # (B, 3)
    pred_sorted_labels_top5 = pred_sorted_labels[:, -5:]  # (B, 5)
    pred_sorted_labels_top10 = pred_sorted_labels[:, -10:]  # (B, 10)

    eval_acc_top1 = accuracy_score(gt_labels, pred_sorted_labels_top1)
    eval_acc_top3 = accuracy_score(gt_labels, pred_sorted_labels_top3)
    eval_acc_top5 = accuracy_score(gt_labels, pred_sorted_labels_top5)
    eval_acc_top10 = accuracy_score(gt_labels, pred_sorted_labels_top10)

    eval_mAP = mAP_score(gt_labels, predict_probs, n_classes=class_number)

    outstr = 'acc@1: %.6f, acc@3: %.6f, acc@5: %.6f, acc@10: %.6f, mAP: %.6f' % \
             (eval_acc_top1 * 100, eval_acc_top3 * 100, eval_acc_top5 * 100, eval_acc_top10 * 100, eval_mAP * 100)
    print(outstr)
