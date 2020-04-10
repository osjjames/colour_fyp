import gen_training_set as gts

# gts.training_set_from_video('/src/data/train_vids/gbh360.mp4', 5)
gts.training_set_from_video('/src/test_vids/vid.mp4', 5, use_csv = True)

# gts.save_images(train_X, '/src/data/train/', 'gbh-')
# gts.save_images(train_y, '/src/data/train/', 'gbh-')



