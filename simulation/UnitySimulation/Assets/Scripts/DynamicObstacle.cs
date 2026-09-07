using UnityEngine;

public class DynamicObstacle : MonoBehaviour
{
    public float speed = 2f;
    public float distance = 5f;

    private Vector3 startPosition;

    void Start()
    {
        startPosition = transform.position;
    }

    void Update()
    {
        float movement = Mathf.Sin(Time.time * speed) * distance;
        transform.position = startPosition + new Vector3(movement, 0f, 0f);
    }
}