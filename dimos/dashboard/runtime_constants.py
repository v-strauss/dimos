import tempfile

def make_constants(json_data):
    import os
    import json
    import psutil
    for each in psutil.Process(os.getpid()).parents():
        try:
            with open(f'/tmp/{each}.json', 'r') as infile:
                return json.load(infile)
        except:
            pass
    # if none of the parents have a json file, make one
    with open(f'/tmp/{os.getpid()}.json', 'w') as outfile:
        json.dump(json_data, outfile)
    return json_data

constants = make_constants(dict(
    default_rerun_grpc_port=9876,
    dashboard_started_lock=tempfile.NamedTemporaryFile(delete=False).name,
))