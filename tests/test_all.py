import test_AvgPool, test_CL, test_Conv, test_ConvTranspose, test_Flatten, test_Linear
import warnings

test_CL.main()
test_AvgPool.main()
test_Conv.main()
#TODO test_ConvTranspose.main()
warnings.warn('test_ConvTranspose is missing because the ConvTranspose modules are not ready.', RuntimeWarning)
test_Flatten.main()
#TODO test_Linear.main()
warnings.warn('test_Linear is missing because the Linear module is not ready.', RuntimeWarning)
