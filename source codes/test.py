"""
    # TODO
    precision_vector = []
    number_of_samples = 0
    while True:
        image_raw, image_name = None
        image = cv2.resize(image_raw, (w, h)).flatten()
        coeff = calc_coefficient(image, mean, pca.components_, k)
        reconstructed = reconstruct_model(mean, pca.components_, coeff)
        error = calc_error(reconstructed, image)
        is_face = detect(error)
        if is_face:
            print('the image', image_name, 'is a face')
            precision_vector.append([1])
        else:
            print('the image', image_name, 'is not a face')
            precision_vector.append([0])
        number_of_samples += 1

        break
    while True:
        image_raw, image_name = None
        image = cv2.resize(image_raw, (w, h)).flatten()
        coeff = calc_coefficient(image, mean, pca.components_, k)
        reconstructed = reconstruct_model(mean, pca.components_, coeff)
        error = calc_error(reconstructed, image)
        is_face = detect(error)
        if is_face:
            print('the image', image_name, 'is a face')
            precision_vector.append([0])
        else:
            print('the image', image_name, 'is not a face')
            precision_vector.append([1])
        number_of_samples += 1
        break
    precision = 100 * np.sum(precision_vector)/precision_vector.__len__()
    print('The model precision for the test data in folder detect is ', precision)
"""
