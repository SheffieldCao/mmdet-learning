import numpy as np 
import mmcv

class DepthEval:

    def load_gts(self, gt_paths, file_client):
        gtList = []
        if file_client is None:
            file_client = mmcv.FileClient(backend='disk')
        for gt_path in gt_paths:
            img_bytes = file_client.get(gt_path)
            gt = mmcv.imfrombytes(img_bytes, flag='unchanged').squeeze() / 256.0
            gtList.append(gt)

        return gtList

    def _compute_errors(self, gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        d1 = (thresh < 1.25).mean()
        d2 = (thresh < 1.25 ** 2).mean()
        d3 = (thresh < 1.25 ** 3).mean()
        
        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())
        
        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())
        
        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)
        
        err = np.log(pred) - np.log(gt)
        silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
        
        err = np.abs(np.log10(pred) - np.log10(gt))
        log10 = np.mean(err)
        
        return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3

    def evaluateImgList(self, preds, gts, min_depth_eval=1e-3, max_depth_eval=80):
        # evaluate the result list
        assert len(preds) == len(gts), "Preds num do not match gts num"
        num_samples = len(preds)
        silog = np.zeros(num_samples, np.float32)
        log10 = np.zeros(num_samples, np.float32)
        rms = np.zeros(num_samples, np.float32)
        log_rms = np.zeros(num_samples, np.float32)
        abs_rel = np.zeros(num_samples, np.float32)
        sq_rel = np.zeros(num_samples, np.float32)
        d1 = np.zeros(num_samples, np.float32)
        d2 = np.zeros(num_samples, np.float32)
        d3 = np.zeros(num_samples, np.float32)

        for i in range(num_samples):
            gt_depth = gts[i]
            pred_depth = preds[i]
            
            if self.do_kitti_benchmark_crop:
                height, width = gt_depth.shape
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
                pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
                pred_depth = pred_depth_uncropped
            
            pred_depth[pred_depth < min_depth_eval] = min_depth_eval
            pred_depth[pred_depth > max_depth_eval] = max_depth_eval
            pred_depth[np.isinf(pred_depth)] = max_depth_eval
            pred_depth[np.isnan(pred_depth)] = min_depth_eval
            
            valid_mask = np.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)
            
            # if args.garg_crop or args.eigen_crop:
            #     gt_height, gt_width = gt_depth.shape
            #     eval_mask = np.zeros(valid_mask.shape)
                
            #     if args.garg_crop:
            #         eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
                
            #     elif args.eigen_crop:
            #         if args.dataset == 'kitti':
            #             eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            #         else:
            #             eval_mask[45:471, 41:601] = 1
                
            #     valid_mask = np.logical_and(valid_mask, eval_mask)
            
            silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i] = self._compute_errors(
                gt_depth[valid_mask], pred_depth[valid_mask])
        
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                    'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
        print("{:7.4f}, {:7.4f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(
            silog.mean(), abs_rel.mean(), log10.mean(), rms.mean(), sq_rel.mean(), log_rms.mean(), d1.mean(), d2.mean(),
            d3.mean()))
        
        return dict(silog=silog, log10=log10, abs_rel=abs_rel, log_rms=log_rms, sq_rel=sq_rel, log_rms=log_rms, d1=d1, d2=d2, d3=d3)

    def evaluateDepth(self, results, gtImageList, file_client):
        # get gts
        gtList = self.load_gts(gtImageList, file_client)

        # eval results only support list[np.ndarray]
        if isinstance(results[0], np.ndarray):
            eval_res = self.evaluateImgList(results, gtList)
        elif isinstance(results, dict) and 'depth_result' in results:
            eval_res = self.evaluateImgList(results['depth_result'], gtList)
        else:
            raise TypeError('Pred results should be a list of numpy.ndarray')
        
        return eval_res
