import timeit

print(timeit.timeit('convert_states()', setup='from test_image_reshape import convert_states', number=1000))
print(timeit.timeit('image_convert()', setup='from test_image_reshape import image_convert', number=1000))
