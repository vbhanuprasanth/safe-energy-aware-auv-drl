using UnityEngine;

public class AUVCollisionDetector : MonoBehaviour
{
    private void OnCollisionEnter(Collision collision)
    {
        Debug.Log("AUV COLLISION with: " + collision.gameObject.name);

        // Obstacle collision
        if (collision.gameObject.name == "Obstacle_01" ||
            collision.gameObject.name == "Obstacle_02" ||
            collision.gameObject.name == "DynamicObstacle_01")
        {
            Debug.Log("AUV HIT OBSTACLE - EPISODE ENDED");

            AUVAgent agent = GetComponent<AUVAgent>();

            if (agent != null)
            {
                agent.AddReward(-250f);
                agent.EndEpisode();
            }
        }
        else if (collision.gameObject.name == "SimulationFloor")
        {
            Debug.Log("AUV HIT FLOOR - EPISODE ENDED");

            AUVAgent agent = GetComponent<AUVAgent>();

            if (agent != null)
            {
                agent.AddReward(-100f);
                agent.EndEpisode();
            }
        }
    }
}