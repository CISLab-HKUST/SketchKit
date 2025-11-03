from sketchkit.animation import Animator

if __name__ == '__main__':
    animator = Animator()

    img0 = "9-0.png"
    img1 = "9-1.png"
    animator.run(img0, img1, inner_frames=8)