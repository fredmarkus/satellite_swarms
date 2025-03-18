from gnc_payload.sensors.imu import IMU, IMUNoiseParams
from gnc_payload.sensors.sensor import SensorNoiseParams
from gnc_payload.sensors.bias import BiasParams

def imu_init(dt: float) -> IMU:
    """
    Initializes the IMU.

    :param dt: The time step for the simulation.

    :return: The initialized IMU.
    """
    # Initialize the IMU
    # bias params are min max range of bias and sigma_w
    # [units] and [(units/s)/sqrt(Hz)]
    bias_params_x = BiasParams.get_random_params([-1e-2, 1e-2], [1e-6, 1e-5])
    bias_params_y = BiasParams.get_random_params([-1e-2, 1e-2], [1e-6, 1e-5])
    bias_params_z = BiasParams.get_random_params([-1e-2, 1e-2], [1e-6, 1e-5])
    # bias_params = BiasParams.get_random_params([0, 0], [0, 0])
    # sigma_v [units/sqrt(Hz)] & scale_factor_error [-]
    sensor_noise_params_accel_x = SensorNoiseParams.get_random_params(
        bias_params_x, [0, 0.0], [0, 0.0]
    )
    sensor_noise_params_accel_y = SensorNoiseParams.get_random_params(
        bias_params_y, [0, 0.0], [0, 0.0]
    )
    sensor_noise_params_accel_z = SensorNoiseParams.get_random_params(
        bias_params_z, [0, 0.0], [0, 0.0]
    )
    sensor_noise_params_accel = [
        sensor_noise_params_accel_x,
        sensor_noise_params_accel_y,
        sensor_noise_params_accel_z,
    ]
    # sigma_v [units/sqrt(Hz)] & scale_factor_error [-]
    sensor_noise_params_gyro_x = SensorNoiseParams.get_random_params(
        bias_params_x, [1e-6, 1e-5], [0, 0.01]
    )
    sensor_noise_params_gyro_y = SensorNoiseParams.get_random_params(
        bias_params_y, [1e-6, 1e-5], [0, 0.01]
    )
    sensor_noise_params_gyro_z = SensorNoiseParams.get_random_params(
        bias_params_z, [1e-6, 1e-5], [0, 0.01]
    )
    sensor_noise_params_gyro = [
        sensor_noise_params_gyro_x,
        sensor_noise_params_gyro_y,
        sensor_noise_params_gyro_z,
    ]

    imu_noise_params = IMUNoiseParams(
        gyro_params=sensor_noise_params_gyro, accel_params=sensor_noise_params_accel
    )
    imu = IMU(
        dt=dt,
        IMU_noise_params=imu_noise_params,
        misalignment_range=[-0.01, 0.01],
    )

    return imu
