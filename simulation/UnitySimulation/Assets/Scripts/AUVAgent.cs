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

    [Header("Environment Bounds")]
    [SerializeField]
    private Vector3 environmentMin =
        new Vector3(0f, 0f, 0f);

    [SerializeField]
    private Vector3 environmentMax =
        new Vector3(20f, 20f, 10f);

    [Header("AUV Settings")]
    [SerializeField]
    private float auvRadius = 0.5f;

    [Header("Safety Settings")]
    [SerializeField]
    private float safetyDistance = 2.0f;

    public override void OnEpisodeBegin()
    {
        Vector3 newStartPosition =
            GenerateRandomStartPosition();

        transform.position = newStartPosition;
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

        previousDistanceToGoal =
            Vector3.Distance(
                transform.position,
                goal.position
            );
    }

    private Vector3 GenerateRandomStartPosition()
    {
        for (int attempt = 0; attempt < 100; attempt++)
        {
            Vector3 position = new Vector3(
                Random.Range(1f, 19f),
                Random.Range(1f, 19f),
                Random.Range(1f, 9f)
            );

            // Keep sufficiently far from goal
            float distanceToGoal =
                Vector3.Distance(
                    position,
                    goal.position
                );

            if (distanceToGoal < 5f)
            {
                continue;
            }

            bool validPosition = true;

            foreach (Transform obstacle in obstacles)
            {
                if (obstacle == null)
                {
                    continue;
                }

                float obstacleRadius =
                    GetObstacleRadius(obstacle);

                float distance =
                    Vector3.Distance(
                        position,
                        obstacle.position
                    );

                float minimumSafeDistance =
                    auvRadius
                    + obstacleRadius
                    + safetyDistance;

                if (distance < minimumSafeDistance)
                {
                    validPosition = false;
                    break;
                }
            }

            if (validPosition)
            {
                return position;
            }
        }

        // Fallback start position
        return startPosition;
    }

    private float GetObstacleRadius(
        Transform obstacle
    )
    {
        SphereCollider sphereCollider =
            obstacle.GetComponent<SphereCollider>();

        if (sphereCollider != null)
        {
            float maximumScale = Mathf.Max(
                obstacle.lossyScale.x,
                obstacle.lossyScale.y,
                obstacle.lossyScale.z
            );

            return sphereCollider.radius *
                   maximumScale;
        }

        return 1.0f;
    }

    public override void CollectObservations(
        VectorSensor sensor
    )
    {
        Vector3 environmentSize =
            new Vector3(20f, 20f, 10f);

        float maxDistance = 30f;

        // ==========================================
        // 1. AUV POSITION (3)
        // ==========================================

        Vector3 normalizedPosition =
            new Vector3(
                transform.position.x /
                    environmentSize.x,

                transform.position.y /
                    environmentSize.y,

                transform.position.z /
                    environmentSize.z
            );

        sensor.AddObservation(
            normalizedPosition
        );

        // ==========================================
        // 2. RELATIVE POSITION TO GOAL (3)
        // ==========================================

        Vector3 relativeGoal =
            goal.position - transform.position;

        Vector3 normalizedGoal =
            new Vector3(
                relativeGoal.x /
                    environmentSize.x,

                relativeGoal.y /
                    environmentSize.y,

                relativeGoal.z /
                    environmentSize.z
            );

        sensor.AddObservation(
            normalizedGoal
        );

        // ==========================================
        // 3. DISTANCE TO GOAL (1)
        // ==========================================

        float distanceToGoal =
            Vector3.Distance(
                transform.position,
                goal.position
            );

        sensor.AddObservation(
            distanceToGoal / maxDistance
        );

        // ==========================================
        // 4. REMAINING ENERGY (1)
        // ==========================================

        float normalizedEnergy =
            energy.RemainingEnergy /
            energy.initialEnergy;

        sensor.AddObservation(
            normalizedEnergy
        );

        // ==========================================
        // 5. FIVE NEAREST OBSTACLES (20)
        // ==========================================

        Transform[] nearestObstacles =
            new Transform[5];

        float[] nearestDistances =
            new float[5];

        for (int i = 0; i < 5; i++)
        {
            nearestDistances[i] =
                float.MaxValue;
        }

        foreach (Transform obstacle in obstacles)
        {
            if (obstacle == null)
            {
                continue;
            }

            float distance =
                Vector3.Distance(
                    transform.position,
                    obstacle.position
                );

            for (int i = 0; i < 5; i++)
            {
                if (distance < nearestDistances[i])
                {
                    for (int j = 4; j > i; j--)
                    {
                        nearestDistances[j] =
                            nearestDistances[j - 1];

                        nearestObstacles[j] =
                            nearestObstacles[j - 1];
                    }

                    nearestDistances[i] =
                        distance;

                    nearestObstacles[i] =
                        obstacle;

                    break;
                }
            }
        }

        // Add exactly 20 obstacle observations
        for (int i = 0; i < 5; i++)
        {
            if (nearestObstacles[i] != null)
            {
                Vector3 relativeObstacle =
                    nearestObstacles[i].position
                    - transform.position;

                Vector3 normalizedObstacle =
                    new Vector3(
                        relativeObstacle.x /
                            environmentSize.x,

                        relativeObstacle.y /
                            environmentSize.y,

                        relativeObstacle.z /
                            environmentSize.z
                    );

                // Relative X, Y, Z
                sensor.AddObservation(
                    normalizedObstacle
                );

                // Distance
                sensor.AddObservation(
                    nearestDistances[i] /
                    maxDistance
                );
            }
            else
            {
                sensor.AddObservation(
                    Vector3.zero
                );

                sensor.AddObservation(
                    0f
                );
            }
        }
    }

    private bool IsOutsideEnvironment()
    {
        Vector3 position =
            transform.position;

        return
            position.x < environmentMin.x ||
            position.x > environmentMax.x ||
            position.y < environmentMin.y ||
            position.y > environmentMax.y ||
            position.z < environmentMin.z ||
            position.z > environmentMax.z;
    }

    private bool CheckSafetyViolation()
    {
        foreach (Transform obstacle in obstacles)
        {
            if (obstacle == null)
            {
                continue;
            }

            float obstacleRadius =
                GetObstacleRadius(obstacle);

            float distance =
                Vector3.Distance(
                    transform.position,
                    obstacle.position
                );

            float safeDistance =
                auvRadius
                + obstacleRadius
                + safetyDistance;

            if (distance < safeDistance)
            {
                return true;
            }
        }

        return false;
    }

    public override void OnActionReceived(
        ActionBuffers actions
    )
    {
        // ==========================================
        // ACTIONS
        // ==========================================

        float actionX = Mathf.Clamp(
            actions.ContinuousActions[0],
            -1f,
            1f
        );

        float actionY = Mathf.Clamp(
            actions.ContinuousActions[1],
            -1f,
            1f
        );

        float actionZ = Mathf.Clamp(
            actions.ContinuousActions[2],
            -1f,
            1f
        );

        Vector3 action =
            new Vector3(
                actionX,
                actionY,
                actionZ
            );

        // ==========================================
        // ENERGY CONSUMPTION
        // ==========================================

        float energyConsumed =
            0.5f * action.magnitude;

        if (energy != null)
        {
            energy.ConsumeEnergy(action);
        }

        // ==========================================
        // MOVE AUV
        // ==========================================

        if (rb != null)
        {
            rb.MovePosition(
                rb.position + action
            );
        }

        // ==========================================
        // BOUNDARY CHECK
        // ==========================================

        bool hitBoundary =
            IsOutsideEnvironment();

        // ==========================================
        // DISTANCE TO GOAL
        // ==========================================

        float currentDistanceToGoal =
            Vector3.Distance(
                transform.position,
                goal.position
            );

        float distanceProgress =
            previousDistanceToGoal
            - currentDistanceToGoal;

        // ==========================================
        // REWARD COMPONENTS
        // ==========================================

        float progressReward =
            distanceProgress * 20f;

        float energyPenalty =
            energyConsumed;

        float stepPenalty =
            -0.2f;

        // ==========================================
        // SAFETY CHECK
        // ==========================================

        float safetyPenalty = 0f;

        if (CheckSafetyViolation())
        {
            safetyPenalty = -25f;
        }

        // ==========================================
        // BOUNDARY PENALTY
        // ==========================================

        float boundaryPenalty =
            hitBoundary ? -100f : 0f;

        // ==========================================
        // TOTAL REWARD
        // ==========================================

        float reward =
            progressReward
            - energyPenalty
            + stepPenalty
            + safetyPenalty
            + boundaryPenalty;

        AddReward(reward);

        // ==========================================
        // UPDATE PREVIOUS DISTANCE
        // ==========================================

        previousDistanceToGoal =
            currentDistanceToGoal;

        // ==========================================
        // END IF BOUNDARY HIT
        // ==========================================

        if (hitBoundary)
        {
            EndEpisode();
            return;
        }

        // ==========================================
        // ENERGY DEPLETION
        // ==========================================

        if (energy != null &&
            energy.RemainingEnergy <= 0f)
        {
            EndEpisode();
        }
    }

    public override void Heuristic(
        in ActionBuffers actionsOut
    )
    {
        float horizontal =
            Input.GetAxis("Horizontal");

        float vertical =
            Input.GetAxis("Vertical");

        float upDown = 0f;

        if (Input.GetKey(KeyCode.E))
        {
            upDown = 1f;
        }
        else if (Input.GetKey(KeyCode.Q))
        {
            upDown = -1f;
        }

        var continuousActions =
            actionsOut.ContinuousActions;

        continuousActions[0] =
            horizontal;

        continuousActions[1] =
            upDown;

        continuousActions[2] =
            vertical;
    }

    // ==========================================================
    // PPO BRIDGE: GET CURRENT 28 OBSERVATIONS
    // ==========================================================

    public float[] GetCurrentObservations()
    {
        Vector3 environmentSize =
            new Vector3(20f, 20f, 10f);

        float maxDistance = 30f;

        float[] observations = new float[28];

        int index = 0;

        // 1. AUV POSITION (3)
        Vector3 normalizedPosition =
            new Vector3(
                transform.position.x / environmentSize.x,
                transform.position.y / environmentSize.y,
                transform.position.z / environmentSize.z
            );

        observations[index++] = normalizedPosition.x;
        observations[index++] = normalizedPosition.y;
        observations[index++] = normalizedPosition.z;

        // 2. RELATIVE POSITION TO GOAL (3)
        Vector3 relativeGoal =
            goal.position - transform.position;

        Vector3 normalizedGoal =
            new Vector3(
                relativeGoal.x / environmentSize.x,
                relativeGoal.y / environmentSize.y,
                relativeGoal.z / environmentSize.z
            );

        observations[index++] = normalizedGoal.x;
        observations[index++] = normalizedGoal.y;
        observations[index++] = normalizedGoal.z;

        // 3. DISTANCE TO GOAL (1)
        float distanceToGoal =
            Vector3.Distance(
                transform.position,
                goal.position
            );

        observations[index++] =
            distanceToGoal / maxDistance;

        // 4. REMAINING ENERGY (1)
        float normalizedEnergy =
            energy.RemainingEnergy /
            energy.initialEnergy;

        observations[index++] =
            normalizedEnergy;

        // 5. FIVE NEAREST OBSTACLES (20)
        Transform[] nearestObstacles =
            new Transform[5];

        float[] nearestDistances =
            new float[5];

        for (int i = 0; i < 5; i++)
        {
            nearestDistances[i] =
                float.MaxValue;
        }

        foreach (Transform obstacle in obstacles)
        {
            if (obstacle == null)
            {
                continue;
            }

            float distance =
                Vector3.Distance(
                    transform.position,
                    obstacle.position
                );

            for (int i = 0; i < 5; i++)
            {
                if (distance < nearestDistances[i])
                {
                    for (int j = 4; j > i; j--)
                    {
                        nearestDistances[j] =
                            nearestDistances[j - 1];

                        nearestObstacles[j] =
                            nearestObstacles[j - 1];
                    }

                    nearestDistances[i] =
                        distance;

                    nearestObstacles[i] =
                        obstacle;

                    break;
                }
            }
        }

        for (int i = 0; i < 5; i++)
        {
            if (nearestObstacles[i] != null)
            {
                Vector3 relativeObstacle =
                    nearestObstacles[i].position
                    - transform.position;

                Vector3 normalizedObstacle =
                    new Vector3(
                        relativeObstacle.x /
                            environmentSize.x,

                        relativeObstacle.y /
                            environmentSize.y,

                        relativeObstacle.z /
                            environmentSize.z
                    );

                observations[index++] =
                    normalizedObstacle.x;

                observations[index++] =
                    normalizedObstacle.y;

                observations[index++] =
                    normalizedObstacle.z;

                observations[index++] =
                    nearestDistances[i] /
                    maxDistance;
            }
            else
            {
                observations[index++] = 0f;
                observations[index++] = 0f;
                observations[index++] = 0f;
                observations[index++] = 0f;
            }
        }

        return observations;
    }


    // ==========================================================
    // PPO BRIDGE: APPLY 3 PPO ACTIONS
    // ==========================================================

    public void ApplyExternalAction(float[] actions)
    {
        if (actions == null || actions.Length != 3)
        {
            Debug.LogError(
                "PPO BRIDGE: Expected exactly 3 actions."
            );

            return;
        }

        float actionX =
            Mathf.Clamp(actions[0], -1f, 1f);

        float actionY =
            Mathf.Clamp(actions[1], -1f, 1f);

        float actionZ =
            Mathf.Clamp(actions[2], -1f, 1f);

        Vector3 action =
            new Vector3(
                actionX,
                actionY,
                actionZ
            );

        // Energy consumption
        float energyConsumed =
            0.5f * action.magnitude;

        if (energy != null)
        {
            energy.ConsumeEnergy(action);
        }

        // Move AUV
        if (rb != null)
        {
            rb.MovePosition(
                rb.position + action
            );
        }

        // Boundary check
        bool hitBoundary =
            IsOutsideEnvironment();

        // Distance to goal
        float currentDistanceToGoal =
            Vector3.Distance(
                transform.position,
                goal.position
            );

        float distanceProgress =
            previousDistanceToGoal
            - currentDistanceToGoal;

        // Reward
        float progressReward =
            distanceProgress * 20f;

        float energyPenalty =
            energyConsumed;

        float stepPenalty =
            -0.2f;

        float safetyPenalty = 0f;

        if (CheckSafetyViolation())
        {
            safetyPenalty = -25f;
        }

        float boundaryPenalty =
            hitBoundary ? -100f : 0f;

        float reward =
            progressReward
            - energyPenalty
            + stepPenalty
            + safetyPenalty
            + boundaryPenalty;

        AddReward(reward);

        previousDistanceToGoal =
            currentDistanceToGoal;

        // Boundary termination
        if (hitBoundary)
        {
            EndEpisode();
            return;
        }

        // Energy termination
        if (energy != null &&
            energy.RemainingEnergy <= 0f)
        {
            EndEpisode();
        }
    }

}