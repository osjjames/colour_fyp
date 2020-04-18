# import gen_training_set as gts
# import histogram_test as ht
# import histogram_better as hb
import time

import cnn

t = time.time()
# gts.training_set_from_video('/src/data/train_vids/gbh360.mp4', 5)
cnn.create()
# gts.training_set_from_video('/src/test_vids/vid.mp4', 5, use_csv = True)
# hb.split_video('/src/data/train_vids/grand_budapest_hotel.mp4', show_cuts = True, save_to_csv = True)


# gts.save_images(train_X, '/src/data/train/', 'gbh-')
# gts.save_images(train_y, '/src/data/train/', 'gbh-')
elapsed = time.time() - t
print(elapsed)


