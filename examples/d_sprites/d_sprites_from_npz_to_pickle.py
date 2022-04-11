import numpy as np
import torch


if __name__ == '__main__':

    # Load dataset
    file_name = "./dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    dataset = np.load(file_name, allow_pickle=True, encoding='latin1')

    # Convert from numpy to pytorch
    images = torch.from_numpy(dataset["imgs"])

    # Save dataset
    nb_chucks = 10
    slice_size = int(images.size(0) / nb_chucks)
    for i in range(nb_chucks):
        ith_images_block = images[slice_size*i:slice_size*(i+1), :, :]
        torch.save(ith_images_block.clone(), "images" + str(i) + ".pt", _use_new_zipfile_serialization=True)

    #
    # Below are some exploratory prints and the associated console output.
    #

    # print(dataset["imgs"].shape)\
    # (737280, 64, 64)

    # print(dataset["latents_values"].shape)
    # (737280, 6)

    # print(dataset["latents_values"][0])
    # [1.  1.  0.5 0.  0.  0. ]

    # print(dataset["latents_classes"].shape)
    # (737280, 6)

    # print(dataset["latents_classes"][0])
    # [0 0 0 0 0 0]

    # print(dataset["metadata"])
    # {
    #     'date': 'April 2017',
    #     'description': 'Disentanglement test Sprites dataset.Procedurally generated 2D shapes, from 6 disentangled'
    #                    'latent factors.This dataset uses 6 latents, controlling the color, shape, scale, rotation and'
    #                    'position of a sprite. All possible variations of the latents are present. Ordering along'
    #                    'dimension 1 is fixed and can be mapped back to the exact latent values that generated that'
    #                    'image.We made sure that the pixel outputs are different. No noise added.',
    #     'version': 1,
    #     'latents_names': ('color', 'shape', 'scale', 'orientation', 'posX', 'posY'),
    #     'latents_possible_values': {
    #         'orientation': [0., 0.16110732, 0.32221463, 0.48332195, 0.64442926,0.80553658, 0.96664389, 1.12775121,
    #                         1.28885852, 1.44996584, 1.61107316, 1.77218047, 1.93328779, 2.0943951 , 2.25550242,
    #                         2.41660973, 2.57771705, 2.73882436, 2.89993168, 3.061039, 3.22214631, 3.38325363,
    #                         3.54436094, 3.70546826, 3.86657557, 4.02768289, 4.1887902 , 4.34989752, 4.51100484,
    #                         4.67211215, 4.83321947, 4.99432678, 5.1554341 , 5.31654141, 5.47764873, 5.63875604,
    #                         5.79986336, 5.96097068, 6.12207799, 6.28318531],
    #         'posX': [0., 0.03225806, 0.06451613, 0.09677419, 0.12903226, 0.16129032, 0.19354839, 0.22580645,
    #                  0.25806452, 0.29032258, 0.32258065, 0.35483871, 0.38709677, 0.41935484, 0.4516129, 0.48387097,
    #                  0.51612903, 0.5483871 , 0.58064516, 0.61290323, 0.64516129, 0.67741935, 0.70967742, 0.74193548,
    #                  0.77419355, 0.80645161, 0.83870968, 0.87096774, 0.90322581, 0.93548387, 0.96774194, 1.],
    #         'posY': [0., 0.03225806, 0.06451613, 0.09677419, 0.12903226, 0.16129032, 0.19354839, 0.22580645,
    #                  0.25806452, 0.29032258, 0.32258065, 0.35483871, 0.38709677, 0.41935484, 0.4516129, 0.48387097,
    #                  0.51612903, 0.5483871 , 0.58064516, 0.61290323, 0.64516129, 0.67741935, 0.70967742,
    #                  0.74193548, 0.77419355, 0.80645161, 0.83870968, 0.8709677, 0.9032258, 0.9354838, 0.9677419, 1.],
    #         'scale': [0.5, 0.6, 0.7, 0.8, 0.9, 1.],
    #         'shape': [1., 2., 3.],
    #         'color': [1.]
    #     },
    #     'latents_sizes': [1, 3, 6, 40, 32, 32],
    #     'author': 'lmatthey@google.com',
    #     'title': 'dSprites dataset'
    # }
