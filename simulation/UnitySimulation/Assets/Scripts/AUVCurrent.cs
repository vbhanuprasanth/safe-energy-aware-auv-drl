using UnityEngine;

public class AUVCurrent : MonoBehaviour
{
    public float currentStrength = 0.5f;

    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    void FixedUpdate()
    {
        Vector3 currentForce = new Vector3(currentStrength, 0f, 0f);

        rb.AddForce(currentForce, ForceMode.Force);
    }
}