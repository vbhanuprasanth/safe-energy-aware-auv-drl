using UnityEngine;

public class GoalDetection : MonoBehaviour
{
    private void OnTriggerEnter(Collider other)
    {
        if (!other.CompareTag("Player"))
            return;

        AUVAgent agent = other.GetComponent<AUVAgent>();

        if (agent != null)
        {
            Debug.Log("GOAL REACHED!");

            agent.AddReward(250f);
            agent.EndEpisode();
        }
    }
}