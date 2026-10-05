using UnityEngine;

public class AUVMetrics : MonoBehaviour
{
    [Header("References")]
    public AUVAgent auvAgent;
    public AUVEnergy energy;

    // Current episode
    private float episodeStartTime;
    private float pathLength;
    private Vector3 previousPosition;
    private bool episodeActive;

    // Overall results
    public int totalEpisodes { get; private set; }
    public int successfulEpisodes { get; private set; }
    public int obstacleCollisions { get; private set; }
    public int dynamicObstacleCollisions { get; private set; }
    public int floorCollisions { get; private set; }
    public int boundaryFailures { get; private set; }
    public int energyFailures { get; private set; }

    public float totalEnergyConsumed { get; private set; }
    public float totalPathLength { get; private set; }
    public float totalNavigationTime { get; private set; }

    private void Awake()
    {
        if (auvAgent == null)
        {
            auvAgent = GetComponent<AUVAgent>();
        }

        if (energy == null)
        {
            energy = GetComponent<AUVEnergy>();
        }
    }

    public void BeginEpisode()
    {
        episodeStartTime = Time.time;
        pathLength = 0f;

        if (auvAgent != null)
        {
            previousPosition = auvAgent.transform.position;
        }

        episodeActive = true;
    }

    private void FixedUpdate()
    {
        if (!episodeActive || auvAgent == null)
        {
            return;
        }

        Vector3 currentPosition =
            auvAgent.transform.position;

        pathLength += Vector3.Distance(
            previousPosition,
            currentPosition
        );

        previousPosition = currentPosition;
    }

    public void RegisterSuccess()
    {
        if (!episodeActive)
        {
            return;
        }

        successfulEpisodes++;

        FinishEpisode("SUCCESS");

    }

    public void RegisterCollision(string objectName)
    {
        if (!episodeActive)
        {
            return;
        }

        if (objectName == "SimulationFloor")
        {
            floorCollisions++;
        }
        else
        {
            obstacleCollisions++;

            if (objectName == "obstacle_06")
            {
                dynamicObstacleCollisions++;
            }
        }

        FinishEpisode(
            "COLLISION: " + objectName
        );
    }

    public void RegisterBoundaryFailure()
    {
        if (!episodeActive)
        {
            return;
        }

        boundaryFailures++;

        FinishEpisode("BOUNDARY");
    }

    public void RegisterEnergyFailure()
    {
        if (!episodeActive)
        {
            return;
        }

        energyFailures++;

        FinishEpisode("ENERGY DEPLETED");
    }

    private void FinishEpisode(string reason)
    {
        if (!episodeActive)
        {
            return;
        }

        float episodeTime =
            Time.time - episodeStartTime;

        float energyConsumed = 0f;

        if (energy != null)
        {
            energyConsumed =
                energy.initialEnergy -
                energy.RemainingEnergy;
        }

        totalEpisodes++;

        totalEnergyConsumed += energyConsumed;
        totalPathLength += pathLength;
        totalNavigationTime += episodeTime;

        episodeActive = false;

        Debug.Log(
            "=== EPISODE RESULT ===\n" +
            "Result: " + reason + "\n" +
            "Episode Time: " +
            episodeTime.ToString("F2") + " s\n" +
            "Path Length: " +
            pathLength.ToString("F2") + "\n" +
            "Energy Consumed: " +
            energyConsumed.ToString("F2") + "\n" +
            "======================"
        );

        PrintOverallResults();
    }

    public void PrintOverallResults()
    {
        float successRate = 0f;
        float collisionRate = 0f;

        if (totalEpisodes > 0)
        {
            successRate =
                (float)successfulEpisodes /
                totalEpisodes * 100f;

            collisionRate =
                (float)obstacleCollisions /
                totalEpisodes * 100f;
        }

        Debug.Log(
            "=== PPO EVALUATION ===\n" +
            "Episodes: " + totalEpisodes + "\n" +
            "Successes: " + successfulEpisodes + "\n" +
            "Obstacle Collisions: " +
                obstacleCollisions + "\n" +
            "Dynamic Obstacle Collisions: " +
                dynamicObstacleCollisions + "\n" +
            "Floor Collisions: " +
                floorCollisions + "\n" +
            "Boundary Failures: " +
                boundaryFailures + "\n" +
            "Energy Failures: " +
                energyFailures + "\n" +
            "Success Rate: " +
                successRate.ToString("F2") + "%\n" +
            "Collision Rate: " +
                collisionRate.ToString("F2") + "%\n" +
            "Total Energy Consumed: " +
                totalEnergyConsumed.ToString("F2") + "\n" +
            "Total Path Length: " +
                totalPathLength.ToString("F2") + "\n" +
            "Total Navigation Time: " +
                totalNavigationTime.ToString("F2") + " s\n" +
            "======================"
        );
    }
}