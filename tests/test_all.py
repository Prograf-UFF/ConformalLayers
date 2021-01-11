import test_AvgPool, test_CL, test_Conv, test_ConvTranspose, test_Flatten, test_Linear
import warnings

test_CL.main()
test_AvgPool.main()
test_Conv.main()
#test_ConvTranspose.main()
warnings.warn('test_ConvTranspose is missing because the ConvTranspose modules are not ready.', RuntimeWarning)
test_Flatten.main()
test_Linear.main()
