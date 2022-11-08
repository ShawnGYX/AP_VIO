import argparse
import csv
from turtle import pos, position
from pylie import R3, SE3, analysis, Trajectory
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------
# Functions
# ---------

def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def computeStatistics(values: list) -> dict:
    # Compute the statistics (mean, std, median, min, max) from a tuple of values
    stats = {}
    stats["rmse"] = float(np.sqrt(np.mean(values**2)))
    stats["mean"] = float(np.mean(values))
    stats["std"] = float(np.std(values))
    stats["med"] = float(np.median(values))
    stats["min"] = float(np.min(values))
    stats["max"] = float(np.max(values))

    return stats


def statString(stats: dict):
    result = ""
    for key, val in stats.items():
        result += "{:>6s}: {:<.4f}\n".format(key, val)
    return result


def read_trajectory(fname: str, tscale: float = 1.0, pose_col: int = 1) -> Trajectory:
    poses = []
    stamps = []
    nanflag = False
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header

        for (i, row) in enumerate(reader):
            stamp = float(row[0]) * tscale
            row = row[pose_col:pose_col+7]
            if stamp < 0:
                continue
            if row==[' ']:
                continue
            try:
                pose = SE3.from_list(row, 'xw')
            except IndexError:
                print("Index error, no points available at t=",stamp)
                pose = poses[-1]
            except ValueError:
                print("NaNs detected in pose data. Approximating output")
                nanflag = True
                pose = poses[-1]

            stamps.append(stamp)
            poses.append(pose)
    return Trajectory(poses, stamps), nanflag


def read_velocities(fname: str) -> Trajectory:
    velocities = []
    stamps = []
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header

        for (i, row) in enumerate(reader):
            stamp = float(row[0])
            row = row[8:11]

            try:
                velocity = R3.from_list(row)
            except ValueError:
                print("NaNs detected in pose data. Ending data here.")
                break

            stamps.append(stamp)
            velocities.append(velocity)
    return Trajectory(velocities, stamps)


# Set argument parser
parser = argparse.ArgumentParser(
    description="Analyse EQVIO performance.")
parser.add_argument("pose_file", metavar='f', type=str,
                    help="The ideal_poses file.")
parser.add_argument("eqvio_file", metavar='e', type=str,
                    help="The eqvio output file.")
parser.add_argument("--fspec", type=str, default='xw',
                    help="The pose formatting. Default xw.")
parser.add_argument("--save", type=str, default=None,
                    help="Save the data to the given location instead of plotting.")
parser.add_argument("--noplot", action="store_true",
                    help="Use this flag to disable plotting.")
parser.add_argument("--tscale", type=float, default=1.0,
                    help="Scaling of time in the ground-truth file. e.g. 1e-9 means the gt timestamps are in ns. Default 1.0.")
args = parser.parse_args()

if args.save is not None:
    if args.save[-1] != "/":
        args.save = args.save + "/"

# Read in the ideal_poses and times
print("Reading the true trajectory from {}".format(args.pose_file))
tru_trajectory, tru_nan_flag = read_trajectory(args.pose_file, args.tscale)

# Read in the EQVIO data
print("Reading EQVIO trajectory from {}".format(args.eqvio_file))
est_trajectory, est_nan_flag = read_trajectory(args.eqvio_file+'IMUState.csv')
est_velocities = read_velocities(args.eqvio_file+'IMUState.csv')
print("reading points")
est_camera_offset, camera_nan_flag = read_trajectory(args.eqvio_file+'points.csv')

nan_flag = tru_nan_flag or est_nan_flag

# Align data
# ----------

print("Aligning the estimated and true states...")

# Check the length of the trajectory
early_finish_flag = (est_trajectory.get_times()[-1] <= 0.9 * (tru_trajectory.get_times(
)[-1] - tru_trajectory.get_times()[0]) + tru_trajectory.get_times()[0])

# Truncate the true data to the EqF
tru_trajectory.truncate(est_trajectory.get_times()[
                        0], est_trajectory.get_times()[-1])
comp_times = est_trajectory.get_times()
tru_trajectory = tru_trajectory[comp_times]
# Get true velocities
tru_velocities = Trajectory([R3(tru_trajectory.get_velocity(
    t)[3:6, :]) for t in comp_times], comp_times)

# Time align the EqF to the true data
est_trajectory = est_trajectory[comp_times]
est_velocities = est_velocities[comp_times]

tru_velocities = np.hstack([vel.as_vector()
                            for vel in tru_velocities.get_elements()])
est_velocities = np.hstack([vel.as_vector()
                            for vel in est_velocities.get_elements()])

# Get true and estimated gravity
gravity_vector = np.array([[0], [0], [-9.81]])
tru_gravity = np.hstack([
    pose.R().inv() * gravity_vector for pose in tru_trajectory.get_elements()])
est_gravity = np.hstack([
    pose.R().inv() * gravity_vector for pose in est_trajectory.get_elements()])

# Align the estimated trajectory to the true
est_trajectory, S = analysis.align_trajectory(
    est_trajectory, tru_trajectory, ret_params=True)

est_positions = np.hstack([pose.x().as_vector()
                           for pose in est_trajectory.get_elements()])
tru_positions = np.hstack([pose.x().as_vector()
                           for pose in tru_trajectory.get_elements()])

#calculate the flight length
position_difference_xyz = tru_positions.T[:-1] - tru_positions.T[1:]
position_difference = np.array([np.sqrt(row[0]**2+row[1]**2+row[2]**2) for row in position_difference_xyz])
flight_length = np.sum(position_difference)

# --------------------
# Print out statistics
# --------------------

ideal_attitude = np.hstack(
    [np.reshape(pose.R()._rot.as_euler("xyz", degrees=True), (3, 1)) for pose in tru_trajectory.get_elements()])
eqvio_attitude = np.hstack(
    [np.reshape(pose.R()._rot.as_euler("xyz", degrees=True), (3, 1)) for pose in est_trajectory.get_elements()])

error_attitude = np.hstack(
    [np.reshape((epose.R() * ipose.R().inv()).log(), (3, 1))
     for (epose, ipose) in zip(est_trajectory.get_elements(), tru_trajectory.get_elements())]
) * 180.0 / np.pi
error_positions = est_positions - tru_positions
error_velocities = est_velocities - tru_velocities

horizontal_velocity_error_stats = computeStatistics(
    np.linalg.norm(error_velocities[:-1], axis=0))

vertical_velocity_error_stats = computeStatistics(error_velocities[-1])

# Absolute position error
absolute_position_error_stats = computeStatistics(
    np.linalg.norm(error_positions, axis=0))

# Absolute attitude error
absolute_attitude_error_stats = computeStatistics(
    np.linalg.norm(error_attitude, axis=0))

# Velocity error
velocity_error_stats = computeStatistics(
    np.linalg.norm(error_velocities, axis=0))

if args.save is None:
    print("Absolute position error (m) stats:")
    print(statString(absolute_position_error_stats))
    print("Absolute attitude error (deg) stats:")
    print(statString(absolute_attitude_error_stats))
    print("Velocity error (m/s) stats:")
    print(statString(velocity_error_stats))
    print("Horizontal Velocity error (m/s) stats:")
    print(statString(horizontal_velocity_error_stats))
    print("vertical Velocity error (m/s) stats:")
    print(statString(vertical_velocity_error_stats))
    print("Scale Error: {:<.4f}".format(S.s().as_float()))
    print("\ntotal flight length (m):",round(flight_length,2))
    print("percentage error in distance:",round(absolute_position_error_stats["rmse"]/flight_length*100,2))
else:
    results_dict = {"position (m)": absolute_position_error_stats,
                    "attitude (d)": absolute_attitude_error_stats,
                    "velocity (m/s)": velocity_error_stats,
                    "Horizontal velocity (m/s)": horizontal_velocity_error_stats,
                    "vertical velocity (m/s)": vertical_velocity_error_stats,
                    "scale": S.s().as_float(),
                    "NaN flag": nan_flag,
                    "Early Finish flag": early_finish_flag,}
    with open(args.save+"results.yaml", 'w') as f:
        yaml.dump(results_dict, f)


# Show some errors
# ----------------

if not args.noplot:

    print("\nPlotting analysis graphs...")

    # -----------------------------------
    # The attitude and position over time
    # -----------------------------------
    fig1, ax = plt.subplots(3, 2)

    # Plot the attitude comparison
    ax[0, 0].set_title("Robot (IMU) Attitude")
    ax[0, 0].plot(comp_times, ideal_attitude[0, :], 'r-')
    ax[0, 0].plot(comp_times, eqvio_attitude[0, :], 'r--')
    ax[0, 0].legend(["True", "Est."])
    ax[0, 0].set_ylabel("Euler x (deg)")
    ax[1, 0].plot(comp_times, ideal_attitude[1, :], 'g-')
    ax[1, 0].plot(comp_times, eqvio_attitude[1, :], 'g--')
    ax[1, 0].set_ylabel("Euler y (deg)")
    ax[2, 0].plot(comp_times, ideal_attitude[2, :], 'b-')
    ax[2, 0].plot(comp_times, eqvio_attitude[2, :], 'b--')
    ax[2, 0].set_ylabel("Euler z (deg)")
    ax[2, 0].set_xlabel("Time (s)")

    ax[0, 0].set_xlim([comp_times[0], comp_times[-1]])
    ax[1, 0].set_xlim([comp_times[0], comp_times[-1]])
    ax[2, 0].set_xlim([comp_times[0], comp_times[-1]])

    # Plot the position comparison
    ax[0, 1].set_title("Robot (IMU) Position")
    ax[0, 1].plot(comp_times, tru_positions[0, :], 'r-')
    ax[0, 1].plot(comp_times, est_positions[0, :], 'r--')
    ax[0, 1].legend(["True", "Est."])
    ax[0, 1].set_ylabel("position x (m)")
    ax[1, 1].plot(comp_times, tru_positions[1, :], 'g-')
    ax[1, 1].plot(comp_times, est_positions[1, :], 'g--')
    ax[1, 1].set_ylabel("position y (m)")
    ax[2, 1].plot(comp_times, tru_positions[2, :], 'b-')
    ax[2, 1].plot(comp_times, est_positions[2, :], 'b--')
    ax[2, 1].set_ylabel("position z (m)")
    ax[2, 1].set_xlabel("Time (s)")

    ax[0, 1].set_xlim([comp_times[0], comp_times[-1]])
    ax[1, 1].set_xlim([comp_times[0], comp_times[-1]])
    ax[2, 1].set_xlim([comp_times[0], comp_times[-1]])

    # ------------------------------------------
    # The attitude and position errors over time
    # ------------------------------------------
    fig2, ax = plt.subplots(4, 2)

    # Plot the attitude error
    ax[0, 0].set_title("Robot Attitude Error")
    ax[0, 0].plot(comp_times, error_attitude[0, :], 'r-')
    ax[0, 0].set_ylabel("Rodrigues x (deg)")
    ax[1, 0].plot(comp_times, error_attitude[1, :], 'g-')
    ax[1, 0].set_ylabel("Rodrigues y (deg)")
    ax[2, 0].plot(comp_times, error_attitude[2, :], 'b-')
    ax[2, 0].set_ylabel("Rodrigues z (deg)")
    ax[3, 0].plot(comp_times, np.linalg.norm(error_attitude, axis=0), 'k-')
    ax[3, 0].set_ylabel("Rodrigues norm (deg)")
    ax[3, 0].set_xlabel("Time (s)")

    ax[0, 0].set_xlim([comp_times[0], comp_times[-1]])
    ax[1, 0].set_xlim([comp_times[0], comp_times[-1]])
    ax[2, 0].set_xlim([comp_times[0], comp_times[-1]])
    ax[3, 0].set_xlim([comp_times[0], comp_times[-1]])
    ax[3, 0].set_ylim([0.0, None])

    # Plot the position error
    ax[0, 1].set_title("Robot Position Error")
    ax[0, 1].plot(comp_times, error_positions[0, :], 'r-')
    ax[0, 1].set_ylabel("x (m)")
    ax[1, 1].plot(comp_times, error_positions[1, :], 'g-')
    ax[1, 1].set_ylabel("y (m)")
    ax[2, 1].plot(comp_times, error_positions[2, :], 'b-')
    ax[2, 1].set_ylabel("z (m)")
    ax[3, 1].plot(comp_times, np.linalg.norm(error_positions, axis=0), 'k-')
    ax[3, 1].set_ylabel("norm (m)")
    ax[3, 1].set_xlabel("Time (s)")

    ax[0, 1].set_xlim([comp_times[0], comp_times[-1]])
    ax[1, 1].set_xlim([comp_times[0], comp_times[-1]])
    ax[2, 1].set_xlim([comp_times[0], comp_times[-1]])
    ax[3, 1].set_xlim([comp_times[0], comp_times[-1]])
    ax[3, 1].set_ylim([0.0, None])

    # ------------------------------------
    # The gravity (and velocity over time)
    # ------------------------------------
    fig3, ax = plt.subplots(3, 2)

    # Plot the gravity comparison
    ax[0, 0].set_title("Gravity Direction")
    ax[0, 0].plot(comp_times, tru_gravity[0, :], 'r-')
    ax[0, 0].plot(comp_times, -est_gravity[0, :], 'r--')
    ax[0, 0].legend(["True", "Est."])
    ax[0, 0].set_ylabel("grav_dir x")
    ax[1, 0].plot(comp_times, tru_gravity[1, :], 'g-')
    ax[1, 0].plot(comp_times, -est_gravity[1, :], 'g--')
    ax[1, 0].set_ylabel("grav_dir y")
    ax[2, 0].plot(comp_times, tru_gravity[2, :], 'b-')
    ax[2, 0].plot(comp_times, -est_gravity[2, :], 'b--')
    ax[2, 0].set_ylabel("grav_dir z")
    ax[2, 0].set_xlabel("Time (s)")

    ax[0, 0].set_xlim([comp_times[0], comp_times[-1]])
    ax[1, 0].set_xlim([comp_times[0], comp_times[-1]])
    ax[2, 0].set_xlim([comp_times[0], comp_times[-1]])

    # Plot the velocity comparison
    ax[0, 1].set_title("Body-Fixed Velocity")
    ax[0, 1].plot(comp_times, tru_velocities[0, :], 'r-')
    ax[0, 1].plot(comp_times, est_velocities[0, :], 'r--')
    ax[0, 1].legend(["True", "Est."])
    ax[0, 1].set_ylabel("bff vel x (m/s)")
    ax[1, 1].plot(comp_times, tru_velocities[1, :], 'g-')
    ax[1, 1].plot(comp_times, est_velocities[1, :], 'g--')
    ax[1, 1].set_ylabel("bff vel y (m/s)")
    ax[2, 1].plot(comp_times, tru_velocities[2, :], 'b-')
    ax[2, 1].plot(comp_times, est_velocities[2, :], 'b--')
    ax[2, 1].set_ylabel("bff vel z (m/s)")
    ax[2, 1].set_xlabel("Time (s)")

    ax[0, 1].set_xlim([comp_times[0], comp_times[-1]])
    ax[1, 1].set_xlim([comp_times[0], comp_times[-1]])
    ax[2, 1].set_xlim([comp_times[0], comp_times[-1]])

    # ---------------------------
    # The camera offset over time
    # ---------------------------
    fig4, ax = plt.subplots(3, 2)
    camera_attitude = np.hstack(
        [np.reshape(pose.R()._rot.as_euler("xyz", degrees=True), (3, 1)) for pose in est_camera_offset.get_elements()])
    camera_position = np.hstack(
        [pose.x().as_vector() for pose in est_camera_offset.get_elements()])
    est_times = est_camera_offset.get_times()

    # Plot the camera offset rotation
    ax[0, 0].set_title("Camera Offset Rotation")
    ax[0, 0].plot(est_times, camera_attitude[0, :], 'r-')
    ax[0, 0].set_ylabel("Rodrigues x (deg)")
    ax[1, 0].plot(est_times, camera_attitude[1, :], 'g-')
    ax[1, 0].set_ylabel("Rodrigues y (deg)")
    ax[2, 0].plot(est_times, camera_attitude[2, :], 'b-')
    ax[2, 0].set_ylabel("Rodrigues z (deg)")
    ax[2, 0].set_xlabel("Time (s)")

    ax[0, 0].set_xlim([est_times[0], est_times[-1]])
    ax[1, 0].set_xlim([est_times[0], est_times[-1]])
    ax[2, 0].set_xlim([est_times[0], est_times[-1]])

    # Plot the camera offset position error
    ax[0, 1].set_title("Camera Offset Position")
    ax[0, 1].plot(est_times, 1e3 * camera_position[0, :], 'r-')
    ax[0, 1].set_ylabel("x (mm)")
    ax[1, 1].plot(est_times, 1e3 * camera_position[1, :], 'g-')
    ax[1, 1].set_ylabel("y (mm)")
    ax[2, 1].plot(est_times, 1e3 * camera_position[2, :], 'b-')
    ax[2, 1].set_ylabel("z (mm)")
    ax[2, 1].set_xlabel("Time (s)")

    ax[0, 1].set_xlim([est_times[0], est_times[-1]])
    ax[1, 1].set_xlim([est_times[0], est_times[-1]])
    ax[2, 1].set_xlim([est_times[0], est_times[-1]])

    if not args.save:
        try:
            from mpldatacursor import datacursor
            datacursor()
        except ImportError:
            print("mpldatacursor not found.")

        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        fig4.tight_layout()
        plt.show()

    if args.save is not None:
        if args.save[-1] != "/":
            args.save = args.save + "/"

        fig1.set_size_inches(11.69, 8.27)
        fig2.set_size_inches(11.69, 8.27)
        fig3.set_size_inches(11.69, 8.27)
        fig4.set_size_inches(11.69, 8.27)
        #fig1.savefig(args.save+"_trajectory.pdf")
        #fig2.savefig(args.save+"_trajectory_error.pdf")
        #fig3.savefig(args.save+"_gravity_and_velocity.pdf")
        #fig4.savefig(args.save+"_camera_offset.pdf")

        filename = args.save+"multi.pdf"
        save_multi_image(filename)
