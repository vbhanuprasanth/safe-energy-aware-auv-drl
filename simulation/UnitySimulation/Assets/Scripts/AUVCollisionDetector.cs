using UnityEngine;

public class AUVCollisionDetector : MonoBehaviour
{
    private AUVController controller;
    private Rigidbody rb;

    private Vector3 startPosition;
    private Quaternion startRotation;

    void Start()
    {
        controller = GetComponent<AUVController>();
        rb = GetComponent<Rigidbody>();

        startPosition = transform.position;
        startRotation = transform.rotation;
    }

    private void OnCollisionEnter(Collision collision)
    {
        Debug.Log("AUV COLLISION with: " + collision.gameObject.name);

        // Goal detection
        if (collision.gameObject.name == "Goal")
        {
            Debug.Log("GOAL REACHED");

            if (controller != null)
            {
                controller.enabled = false;
            }

            Invoke(nameof(ResetEpisode), 2f);
            return;
        }

        // Obstacle collision
        if (collision.gameObject.name == "Obstacle_01" ||
            collision.gameObject.name == "Obstacle_02" ||
            collision.gameObject.name == "DynamicObstacle_01")
        {
            Debug.Log("AUV HIT OBSTACLE - MOVEMENT STOPPED");

            if (controller != null)
            {
                controller.enabled = false;
            }

            Invoke(nameof(ResetEpisode), 2f);
        }
    }

    private void ResetEpisode()
    {
        transform.position = startPosition;
        transform.rotation = startRotation;

        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }

        if (controller != null)
        {
            controller.enabled = true;
        }

        Debug.Log("EPISODE RESET");
    }
}