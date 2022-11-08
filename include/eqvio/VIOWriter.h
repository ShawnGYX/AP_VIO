#pragma once

#include "eqvio/LoopTimer.h"
#include "eqvio/common/aofstream.h"
#include "eqvio/mathematical/VIOState.h"

/** @brief A class to handle writing output for the VIO system
 *
 * The output is distributed over a number of files, which are organised in a given directory. The use of aofstream
 * rather than ofstream means that the VIO system usually does not have to slow down or wait for the OS to write to
 * files.
 */
class VIOWriter {
  protected:
    aofstream IMUStateFile; ///< The file for recording the IMU states
    aofstream cameraFile;   ///< The file for recording the camera offset
    aofstream biasFile;     ///< The file for recording the IMU biases
    aofstream pointsFile;   ///< The file for recording the landmark points

    aofstream landmarkErrorFile; ///< The file for recording landmark depth errors
    aofstream neesFile;          ///< The file for recording NEES

    aofstream featuresFile; ///< The file for recording the image features

    aofstream timingFile;                   ///< The file for recording processing times
    bool timingFileHeaderIsWritten = false; ///< True once the header of the timing file is written.

  public:
    /** @brief Create a number of files in the output directory to record VIO data.
     *
     * @param outputDir The output directory where files should be stored.
     *
     * This method creates the output directory if it does not already exist, and then creates output files for the VIO
     * data to be written to. It also creates header rows in each of the files to make them human-readable.
     */
    void StartVIOWriter(std::string outputDir);

    /** @brief Write the VIO states to the relevant files.
     *
     * @param stamp The timestamp associated with the given state.
     * @param xi The current VIO system state estimate.
     *
     * The VIO states are divided into IMU states, IMU biases, camera offset, and landmark points. All four of these are
     * written to separate output files and time stamped.
     */
    void writeStates(const double& stamp, const VIOState& xi);

    /** @brief Write the feature measurements to file.
     *
     * @param y The vision measurement to be written.
     */
    void writeFeatures(const VisionMeasurement& y);

    /** @brief Write the system processing time to file.
     *
     * @param timingData The timing information for the latest loop of the VIO system.
     *
     * Note that, on the first call to this function, the timing data is also used to create a header for the timing
     * data file.
     */
    void writeTiming(const LoopTimer::LoopTimingData& timingData);

    void writeNEES(const double& stamp, const double& NEES);
    void writeLandmarkError(const double& stamp, const VIOState& trueState, const VIOState& estState);
};