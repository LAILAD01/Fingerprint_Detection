def batch_process(images, labels, batch_size=100):
    num_batches = len(images) // batch_size + (len(images) % batch_size != 0)
    for i in range(num_batches):
        batch_images = images[i * batch_size:(i + 1) * batch_size]
        batch_labels = labels[i * batch_size:(i + 1) * batch_size]
        yield (batch_images, batch_labels)
