import pepi

@pepi.utils.parallelise(callback=pepi.postp, num_processes=4)
def parallel_function(inputs):
    return [process_data(i) for i in inputs]




# with mp.Pool(processes=p-1) as pool:
#     result_objects = [pool.apply_async(pepi_run, (args,)) for args in args_list]
    
#     for result_obj in result_objects:
#         try:
#             result = result_obj.get(timeout=TIMEOUT_DURATION)
#             if result is None:
#                 print("...Missing results, skipping.")
#                 continue

#             else:
#                 id, _, array, time, _, _ = result
#                 result_data = [id, array, time]
#                 print(f"{id}...Saving results.")
#                 results.append(result_data)

#         except mp.context.TimeoutError:
#             print("...Worker timed out, skipping.")
#             continue
