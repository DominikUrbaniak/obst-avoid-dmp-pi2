#!/bin/bash
. install/setup.bash

if [ -z "$1" ]; then
  echo "Usage: $0 <start_index>"
  exit 1
fi

START_ID=$1
JSON_FILE="experiment_list.json"

ros2 service call /high_level_control/go_to_start std_srvs/srv/Empty
ros2 param set /high_level_controller drop_height_offset 0.07
ros2 param set /move_group use_sim_time True
ros2 param set /obstacle_detection visualize_obst_view False
ros2 param set /obstacle_detection visualize_obst_view_all True
sleep 2

TOTAL=$(expr $(jq length "$JSON_FILE") - 1) #last experiment config is in collision at home position
for ((i = START_ID; i < TOTAL; i++)); do
    exp=$(jq ".[$i]" "$JSON_FILE")
    start_x=$(echo "$exp" | jq -r ".start_x")
    start_y=$(echo "$exp" | jq -r ".start_y")
    first_x=$(echo "$exp" | jq -r ".first_x")
    cup_x=$(echo "$exp" | jq -r ".cup_x")
    cup_y=$(echo "$exp" | jq -r ".cup_y")
    ny=$(echo "$exp" | jq -r ".ny")
    nz=$(echo "$exp" | jq -r ".nz")

    echo "Running experiment $i with start_x=$start_x, start_y=$start_y, ny=$ny, nz=$nz, cup_x=$cup_x, cup_y=$cup_y"
    sol_id=-1
    #if (( nz > 2 )); then
    #    sol_id=0
    #else
    #    sol_id=-1
    #fi

    ros2 service call /high_level_control/set_obstacle_config_cup custom_interfaces/srv/BoxWallConfig \
        "{start_pose: {position: {x: $start_x, y: $start_y}}, first_x: $first_x, cup_x: $cup_x, cup_y: $cup_y, ny: 0, nz: 0}"

    neg_start_x=$(echo "-1 * $start_x" | bc)
    neg_start_y=$(echo "-1 * $start_y" | bc)
    ros2 service call /low_level_control/reach_point custom_interfaces/srv/ReachPoint \
        "{x: $neg_start_x, y: $neg_start_y, z: 0.18, qr: 0.0, qp: 3.141592653589793, qy: 1.5707963268}"
    sleep 3

    ros2 service call /high_level_control/set_obstacle_config_cup custom_interfaces/srv/BoxWallConfig \
        "{start_pose: {position: {x: $start_x, y: $start_y}}, first_x: $first_x, cup_x: $cup_x, cup_y: $cup_y, ny: $ny, nz: $nz}"
    sleep 1

    ros2 service call /high_level_control/execute_pick_and_drop custom_interfaces/srv/ExecuteDmpAuto \
        "{obj_dims: [0.07, 0.07, 0.13], sol_id: $sol_id}"
done
