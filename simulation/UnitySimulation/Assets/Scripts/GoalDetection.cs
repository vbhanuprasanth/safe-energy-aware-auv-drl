using UnityEngine;
using System.Collections;

public class GoalDetection : MonoBehaviour
{
    public float resetDelay = 1f;

    private Vector3 startPosition;
    private bool goalReached = false;

    void Start()
    {
        GameObject auv = GameObject.FindGameObjectWithTag("Player");

        if (auv != null)
        {
            startPosition = auv.transform.position;
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (goalReached)
            return;

        if (other.CompareTag("Player"))
        {
            goalReached = true;

            Debug.Log("GOAL REACHED!");

            StartCoroutine(ResetAUV(other.gameObject));
        }
    }

    IEnumerator ResetAUV(GameObject auv)
    {
        yield return new WaitForSeconds(resetDelay);

        Rigidbody rb = auv.GetComponent<Rigidbody>();

        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }

        auv.transform.position = startPosition;

        goalReached = false;

        Debug.Log("AUV RESET TO START");
    }
}