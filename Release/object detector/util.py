
def pixel2relative_position(pixel_position, height, width):
    """
    takes pixel coordinates [0,height_width]^2 and transforms it into relative coordinates [-1,1]^2 of centercropped image
    :pixel_position: tensor of size [batch_size x 2]
    :height_width: height and width of a rectangular image
    :return: tensor of size [batch_size x 2]
    """
    max_h_w = max(height,width)
    pixel_position[:,0] = (2*pixel_position[:,0]+max(height-width,0))/max_h_w-1
    pixel_position[:,1] = (2*pixel_position[:,1]+max(width-height,0))/max_h_w-1
    return pixel_position

def relative2pixel_position(pixel_position, height_width):
    """
    takes relative coordinates [-1,1]^2 and transforms it into pixel coordinates [0,height_width]^2
    :pixel_position: tensor of size [batch_size x 2]
    :height_width: height and width of a rectangular (centercropped) image
    :return: tensor of size [batch_size x 2]
    """
    return (pixel_position+1)*height_width/2
