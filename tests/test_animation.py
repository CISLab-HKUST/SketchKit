from sketchkit.animation import Animator

if __name__ == "__main__":
    animator = Animator()

    img0 = "0-0.png"
    img1 = "0-1.png"
    img_list = animator.run(
        img0, img1, inner_frames=24, output_folder="output/animation"
    )
