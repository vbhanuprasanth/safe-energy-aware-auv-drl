using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class AUVAgent : Agent
{
    private AUVEnergy energy;
    private Rigidbody rb;

    private Vector3 startPosition;
    private Quaternion startRotation;
    private float previousDistanceToGoal;

    protected override void Awake()
    {
        base.Awake();

        energy = GetComponent<AUVEnergy>();
        rb = GetComponent<Rigidbody>();

        startPosition = transform.position;
        startRotation = transform.rotation;
    }

    [Header("Environment References")]
    public Transform goal;
    public Transform[] obstacles;

    public override void OnEpisodeBegin()
    {
        transform.position = startPosition;
        transform.rotation = startRotation;

        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }

        if (energy != null)
        {
            energy.ResetEnergy();
        }

        previousDistanceToGoal = Vector3.Distance(transform.position,goal.position);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Environment normalization size
        Vector3 environmentSize = new Vector3(20f, 20f, 10f);
        float maxDistance = 30f;

        // 1. AUV position (3)
        Vector3 normalizedPosition = new Vector3(
            transform.position.x / environmentSize.x,
            transform.position.y / environmentSize.y,
            transform.position.z / environmentSize.z
        );

        sensor.AddObservation(normalizedPosition);

        // 2. Relative position to goal (3)
        Vector3 relativeGoal = goal.position - transform.position;

        Vector3 normalizedGoal = new Vector3(
            relativeGoal.x / environmentSize.x,
            relativeGoal.y / environmentSize.y,
            relativeGoal.z / environmentSize.z
        );

        sensor.AddObservation(normalizedGoal);

        // 3. Distance to goal (1)
        float distanceToGoal = Vector3.Distance(
            transform.position,
            goal.position
        );

        sensor.AddObservation(distanceToGoal / maxDistance);

        // 4. Remaining energy (1)
        float normalizedEnergy = energy.RemainingEnergy / energy.initialEnergy;

        sensor.AddObservation(normalizedEnergy);

        // 5. Five nearest obstacles (5 × 4 = 20)
        Transform[] nearestObstacles = new Transform[5];
        float[] nearestDistances = new float[5];

        for (int i = 0; i < 5; i++)
        {
            nearestDistances[i] = float.MaxValue;
        }

        foreach (Transform obstacle in obstacles)
        {
            if (obstacle == null)
                continue;

            float distance = Vector3.Distance(
                transform.position,
                obstacle.position
            );

            for (int i = 0; i < 5; i++)
            {
                if (distance < nearestDistances[i])
                {
                    for (int j = 4; j > i; j--)
                    {
                        nearestDistances[j] = nearestDistances[j - 1];
                        nearestObstacles[j] = nearestObstacles[j - 1];
                    }

                    nearestDistances[i] = distance;
                    nearestObstacles[i] = obstacle;

                    break;
                }
            }
        }

        // Add information for exactly 5 obstacles
        for (int i = 0; i < 5; i++)
        {
            if (nearestObstacles[i] != null)
            {
                Vector3 relativeObstacle =
                    nearestObstacles[i].position - transform.position;

                Vector3 normalizedObstacle = new Vector3(
                    relativeObstacle.x / environmentSize.x,
                    relativeObstacle.y / environmentSize.y,
                    relativeObstacle.z / environmentSize.z
                );

                // Relative X, Y, Z
                sensor.AddObservation(normalizedObstacle);

                // Distance
                sensor.AddObservation(
                    nearestDistances[i] / maxDistance
                );
            }
            else
            {
                // Padding when fewer than 5 obstacles exist
                sensor.AddObservation(Vector3.zero);
                sensor.AddObservation(0f);
            }
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float actionX = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float actionY = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        float actionZ = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);

        Vector3 action = new Vector3(actionX, actionY, actionZ);

        // Energy consumption
        float energyConsumed = 0.5f * action.magnitude;

        if (energy != null)
        {
            energy.ConsumeEnergy(action);
        }

        // Move AUV
        if (rb != null)
        {
            rb.MovePosition(
                rb.position + action * Time.fixedDeltaTime
            );
        }

        // Calculate new distance to goal
        float currentDistanceToGoal = Vector3.Distance(
            transform.position,
            goal.position
        );

        // Progress toward goal
        float distanceProgress =
            previousDistanceToGoal - currentDistanceToGoal;

        // Team A reward components
        float progressReward = distanceProgress * 20f;
        float energyPenalty = energyConsumed;
        float stepPenalty = -0.2f;

        // Safety check
        float safetyDistance = 2.0f;
        float safetyPenalty = 0f;

        foreach (Transform obstacle in obstacles)
        {
            if (obstacle == null)
                continue;

            float obstacleDistance =
                Vector3.Distance(transform.position, obstacle.position);

            if (obstacleDistance < safetyDistance)
            {
                safetyPenalty = -25f;
                break;
            }
        }

        // Total reward
        float reward =
            progressReward
            - energyPenalty
            + stepPenalty
            + safetyPenalty;

        AddReward(reward);

        // Update distance for the next action
        previousDistanceToGoal = currentDistanceToGoal;

        // Energy depletion termination
        if (energy != null && energy.RemainingEnergy <= 0f)
        {
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");

        float upDown = 0f;

        if (Input.GetKey(KeyCode.E))
        {
            upDown = 1f;
        }
        else if (Input.GetKey(KeyCode.Q))
        {
            upDown = -1f;
        }

        var continuousActions = actionsOut.ContinuousActions;

        continuousActions[0] = horizontal;
        continuousActions[1] = upDown;
        continuousActions[2] = vertical;
    }
}