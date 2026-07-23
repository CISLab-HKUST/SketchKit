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
    Compute sketch recognition metrics, including ACC@1, ACC@3, ACC@5, ACC@10, and mAP.

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


def segmentation_metrics(gt_labels_list, predict_labels_list, component_correct_threshold=0.75):
    """
    Compute sketch segmentation metrics, including P_metric and C_metric.

    Args:
        gt_labels_list: A list of gt_labels, each for a sketch. Each item includes num_path paths. Each path contains gt labels for points on the path.
        predict_labels_list: (num_sketch, num_point)
        component_correct_threshold: A component/path is correctly labeled if at least 75% of its points have the correct label.

    Returns:

    """
    p_metric_list = []
    c_metric_list = []

    for i in range(len(predict_labels_list)):
        gt_labels = gt_labels_list[i]
        predict_labels = np.array(predict_labels_list[i])  # (num_point)
        predict_labels_left = np.copy(predict_labels)

        n_path = len(gt_labels)

        p_correct = 0
        p_total = len(predict_labels)
        c_correct = 0
        c_total = n_path

        for path_i in range(n_path):
            path_gt_labels = np.array(gt_labels[path_i])  # (num_point_on_path)
            path_predict_labels = predict_labels_left[:len(path_gt_labels)]  # (num_point_on_path)
            predict_labels_left = predict_labels_left[len(path_gt_labels):]

            path_point_correct = np.sum(path_gt_labels == path_predict_labels)
            p_correct += path_point_correct

            patch_label_percent = path_point_correct / float(len(path_gt_labels))
            if patch_label_percent > component_correct_threshold:
                c_correct += 1

        p_metric_list.append(p_correct / float(p_total))
        c_metric_list.append(c_correct / float(c_total))

    p_metric, c_metric = np.average(p_metric_list), np.average(c_metric_list)
    print('P_metric:{:.4}%\tC_metric:{:.4}%'.format(p_metric * 100, c_metric * 100))
