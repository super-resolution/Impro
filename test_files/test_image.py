from processing.Image import microscope_image as image


def test_microscope_image():
    i = image(r"D:\_3dSarahNeu\!!20170317\20170317_0308c_SIM10_Out_Channel Alignment.czi")
    i.parse()
    assert i.data.any()
    assert i.meta_data['SizeX'] != 0
    assert i.meta_data['SizeX'] != 0
    assert i.meta_data['SizeX'] != 0



