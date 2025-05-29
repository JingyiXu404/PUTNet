from setproctitle import setproctitle
import os
setproctitle('test')
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

from evaluator_tool import get_measure
import torch 
print('CSM1') 
# device是torch.device()返回的对象, 返回值 sal_measure是一个字典 
sal_measure1 = get_measure(["max-F","mean-F","MAE","S"],"/media/hpc/data/dm/test_tool/test_data_DUT-O/CSM1/", 
                            "/media/hpc/data/dm/test_tool/test_data_DUT-O/gt/",torch.device('cuda'))
sal_measure2 = get_measure(["max-F","mean-F","MAE","S"],"/media/hpc/data/dm/test_tool/test_data_DUTS-TE/CSM1/",
                             "/media/hpc/data/dm/test_tool/test_data_DUTS-TE/gt/",torch.device('cuda'))
sal_measure3 = get_measure(["max-F","mean-F","MAE","S"], "/media/hpc/data/dm/test_tool/test_data_ECSSD/CSM1/",
                            "/media/hpc/data/dm/test_tool/test_data_ECSSD/gt/",torch.device('cuda'))
sal_measure4 = get_measure(["max-F","mean-F","MAE","S"], "/media/hpc/data/dm/test_tool/test_data_HKU-IS/CSM1/", 
                            "/media/hpc/data/dm/test_tool/test_data_HKU-IS/gt/", torch.device('cuda'))
sal_measure5 = get_measure(["max-F","mean-F","MAE","S"],"/media/hpc/data/dm/test_tool/test_data_Pascal-S/CSM1/", 
                            "/media/hpc/data/dm/test_tool/test_data_Pascal-S/gt/",torch.device('cuda'))

print('CSM1,  jccnet1/jccnet_itr_18000_tar_0.128891.pth')
print('DUT-O',  sal_measure1)
print('DUTS-TE',  sal_measure2)
print('ECSSD',  sal_measure3)
print('HKU-IS',  sal_measure4)
print('Pascal-S',  sal_measure5)