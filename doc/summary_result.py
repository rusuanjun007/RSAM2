from matplotlib import pyplot as plt


def plot_summary_result():
    model_size = [
        "sam2.1_tiny(38.96M)",
        "sam2.1_small(46.06M)",
        "sam2.1_base+(80.85M)",
        "sam2.1_large(224.45M)",
    ]
    original_video_fps = [85.11, 79.99, 60.76, 37.18]
    # sam2_real_time_only = [32.24, 30.93, 27.58, 21.80]
    # sam2_real_time_camera = [29.35, 29.42, 29.17, 22.89]
    # sam2_real_time_camera_viz = [29.31, 29.21, 27.20, 20.95]

    sam2_real_time_only = [32.07, 31.41, 27.99, 21.82]
    sam2_real_time_camera = [21.86, 21.35, 20.05, 16.93]
    sam2_real_time_camera_viz = [20.93, 20.18, 19.0, 16.14]

    plt.figure(figsize=(10, 6))
    plt.plot(model_size, original_video_fps, marker="o", label="Original SAM2.1")
    plt.plot(model_size, sam2_real_time_only, marker="o", label="SAM2.1 Real Time")
    plt.plot(
        model_size,
        sam2_real_time_camera,
        marker="o",
        label="SAM2.1 Real Time + Camera",
    )
    plt.plot(
        model_size,
        sam2_real_time_camera_viz,
        marker="o",
        label="SAM2.1 Real Time + Camera + Viz",
    )

    plt.xlabel("Model Size")
    plt.ylabel("FPS")
    plt.title("FPS vs Model Size | Test On RTX5090")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig("doc/fps_vs_model_size.png")
    plt.show()


if __name__ == "__main__":
    plot_summary_result()
