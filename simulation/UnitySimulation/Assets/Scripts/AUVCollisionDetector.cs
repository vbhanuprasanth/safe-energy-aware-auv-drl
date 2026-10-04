using UnityEngine;

public class AUVCollisionDetector : MonoBehaviour
{
    private void OnCollisionEnter(
        Collision collision
    )
    {
        Debug.Log(
            "AUV COLLISION with: " +
            collision.gameObject.name
        );

        AUVAgent agent =
            GetComponent<AUVAgent>();

        if (agent == null)
            return;

        // =====================================================
        // TEAM A OBSTACLE COLLISION
        // =====================================================

        if (collision.gameObject.name == "obstacle_01" ||
            collision.gameObject.name == "obstacle_02" ||
            collision.gameObject.name == "obstacle_03" ||
            collision.gameObject.name == "obstacle_04" ||
            collision.gameObject.name == "obstacle_05")
        {
            Debug.Log(
                "AUV HIT OBSTACLE - EPISODE ENDED"
            );

            // Team A collision penalty
            agent.AddReward(-250f);

            agent.EndEpisode();

            return;
        }

        // =====================================================
        // ENVIRONMENT BOUNDARY / FLOOR
        // =====================================================

        if (collision.gameObject.name ==
            "SimulationFloor")
        {
            Debug.Log(
                "AUV HIT FLOOR - EPISODE ENDED"
            );

            // Team A boundary penalty
            agent.AddReward(-100f);

            agent.EndEpisode();

            return;
        }
    }
}