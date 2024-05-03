# run script
# read pept data from path
# compute limits, resolution, and efficiency
# create samples
# parallelize the pepi_algorithm
# parallelize the results post_processing
import PEPI
save_path = r"/rds/projects/i/ingrama-unilever-soap-cfd/PEPTpipeline/PEPI/evolved_geom/01"
data_path = r"/rds/projects/i/ingrama-unilever-soap-cfd/PEPTpipeline/PEPI/data/evolved_geom/EvolvedPEPI.da01"
params = utils.create_parameters(
    time_skip=0,
    time_length=1e6,
    time_slice=1,
    overlap=0.5,
    xlim=[112, 491],
    ylim=[47, 556],
    pixel_size=2,
    moving_average=True,
    moving_average_window=5,
    half_life=109.8,
    save_path=save_path)