## WHERE IS THE DATA?!

Running utils.py will prompt you to select **your** raw data file and a directory to store subsequent processed data and graphs. 
The utils.py script generates a JSON source file, data_source, that reads as: 

{"raw_data_file": "/path/to/raw/data.csv", "process_data_dir": "/path/to/process/directory"}

data_source is accessed regularly by subsequent scripts using:
```python
import data.utils
raw_path, process_data_path = data.utils.source_data()
```
This enables the storage of larger data files on external disks etc, and subsequent access across different systems.
Modifications/processed data files are stored in process_data_path/data/,
downloaded and modified OSMNx graphs are stored in process_data_path/graphs/.

[We used taxi data collected in Porto, Portugal, which is available to use here](https://archive.ics.uci.edu/ml/datasets/Taxi+Service+Trajectory+-+Prediction+Challenge,+ECML+PKDD+2015)

---

## WHAT DOES IT LOOK LIKE?!

The raw data file must be a CSV file and look something like this:

| TRIP_ID | TAXI_ID | TIMESTAMP	| MISSING_DATA | POLYLINE |
|:---:    |:---:|:---:      | :---:          | :---:      |
| 1       |   1     | 1399243165 | False | [[-8.593902, 41.139963], [-8.593938, 41.139936], [-8.593992, 41.13999], ...|
| 2 | 2| 1399244365 | False | [[-8.583993, 41.175747], [-8.584092, 41.17527], [-8.584767, 41.173821],...|
| ⋮ | ⋮ | ⋮ | ⋮ |  ⋮ |

The essential columns here are TIMESTAMP and POLYLINE.
* TIMESTAMP: UNIX TIMESTAMP indicating the start of the trajectory.
* POLYLINE: GPS coordinates for each vehicle trajectory. We assume coordinates are equally time-spaced
and in the form above: [[longitude0, latitude0], [longitude1, latitude1], [longitude2, latitude2], ...].
* MISSING_DATA can be used to indicate any polylines with holes in, although these ones will just be discarded.

Of course additional columns will likely be useful for your inference and won't be removed by preprocess.py
or subsequent scripts.

---

## ANYTHING ELSE?!
Yep, just a couple of additional variables that need to be defined in script:

At the top of utils.py, change
`
project_title = 'portotaxi'
`
to whatever you want at the start of generated data files.

At the top of preprocess.py set
`
data_time_step = 15
`
to the number of seconds between the (equally spaced) GPS coordinates.

Set
`
time_zone = timezone('Europe/Lisbon')
`
to the timezone of your data. `pytz.all_timezones` will list available timezones.

Set
`
bbox_ll = [41.176, 41.111, -8.563, -8.657]
`
to define (in longitude-latitude, [N, S, E, W]) a bounding box for inference. Typically inference is focused on
a smaller area than the data covers.

Now run utils.py to set up data_source. Then run preprocess.py to select a subset of the data based on TIMESTAMP
(it is advised to initially work on a small dataset), trim the data to include only coordinates inside the bounding box
and finally convert the data into the UTM format for easier inference (i.e. work on a plane rather than a sphere).


---
## OK, AM I DONE NOW?!
No! Now go do some inference...

And please stop shouting.