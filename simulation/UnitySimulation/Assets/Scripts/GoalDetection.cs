using UnityEngine;

public class GoalDetection : MonoBehaviour
{
    private void OnTriggerEnter(
        Collider other
    )
    {
        if (!other.CompareTag("Player"))
            return;

        AUVAgent agent =
            other.GetComponent<AUVAgent>();

        if (agent == null)
            return;

        Debug.Log(
            "GOAL REACHED!"
        );

        // Team A goal reward
        agent.AddReward(250f);

        agent.EndEpisode();
    }
}